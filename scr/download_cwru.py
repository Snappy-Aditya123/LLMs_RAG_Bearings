#!/usr/bin/env python3
"""Discover, download, organize and index official CWRU bearing MAT files.

pip install requests beautifulsoup4
python download_cwru.py --dry-run
python download_cwru.py
python download_cwru.py --groups normal 12k_de
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://engineering.case.edu"
PAGES = {
    "normal": "/bearingdatacenter/normal-baseline-data",
    "12k_de": "/bearingdatacenter/12k-drive-end-bearing-fault-data",
    "48k_de": "/bearingdatacenter/48k-drive-end-bearing-fault-data",
    "12k_fe": "/bearingdatacenter/12k-fan-end-bearing-fault-data",
}
# These values are approximate tabulated motor speeds, not per-file measurements.
APPROX_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}
FAULT_NAMES = {"IR": "inner_race", "B": "ball", "OR": "outer_race"}
FAULT_LINK = re.compile(r"^(IR|B|OR)(\d{3})(?:@(3|6|12))?_(\d)$", re.IGNORECASE)
NORMAL_LINK = re.compile(r"^Normal_(\d)$", re.IGNORECASE)
COLS = [
    "group", "bearing_end", "sampling_rate_hz", "fault_type", "fault_diameter_in",
    "fault_diameter_mm", "outer_race_position", "motor_load_hp", "approx_motor_speed_rpm",
    "site_label", "original_filename", "local_path", "source_url", "source_page",
    "download_status", "size_bytes",
]


def make_session():
    sess = requests.Session()
    sess.headers.update({"User-Agent": "CWRU-Academic-Dataset-Downloader/1.0 (Python requests)"})
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    return sess


def discover(sess, group):
    page = urljoin(BASE, PAGES[group])
    response = sess.get(page, timeout=(15, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    items = []
    urls = set()
    for a in soup.select("a[href]"):
        label = a.get_text(" ", strip=True)
        url = urljoin(page, a["href"])
        # Only actual .mat files on the university's official host.
        parsed = urlparse(url)
        if parsed.hostname != "engineering.case.edu" or not parsed.path.lower().endswith(".mat"):
            continue
        if url in urls:
            continue
        normal_match = NORMAL_LINK.fullmatch(label)
        fault_match = FAULT_LINK.fullmatch(label)
        if group == "normal" and normal_match:
            load = int(normal_match.group(1))
            fault, diameter, position = "healthy", None, None
        elif group != "normal" and fault_match:
            type_code, mils, position, load_str = fault_match.groups()
            fault = FAULT_NAMES[type_code.upper()]
            diameter = int(mils) / 1000
            load = int(load_str)
            if fault != "outer_race":
                position = None
        else:
            print(f"[WARN] Unrecognized MAT link on {group}: {label!r}", file=sys.stderr)
            continue
        if load not in APPROX_RPM:
            print(f"[WARN] Unexpected motor load: {label}", file=sys.stderr)
            continue
        bearing_end = "fan_end" if group == "12k_fe" else ("none" if group == "normal" else "drive_end")
        sample_rate = "" if group == "normal" else (48000 if group == "48k_de" else 12000)
        # Use group to avoid collisions where the website repeats the same label.
        descriptive_name = f"{label.replace('@', '_at_')}__{Path(unquote(parsed.path)).name}"
        target = Path(group) / fault / descriptive_name
        items.append({
            "group": group, "bearing_end": bearing_end, "sampling_rate_hz": sample_rate,
            "fault_type": fault, "fault_diameter_in": "" if diameter is None else f"{diameter:.3f}",
            "fault_diameter_mm": "" if diameter is None else f"{diameter * 25.4:.4f}",
            "outer_race_position": "" if position is None else f"{position}:00",
            "motor_load_hp": load, "approx_motor_speed_rpm": APPROX_RPM[load],
            "site_label": label, "original_filename": Path(unquote(parsed.path)).name,
            "local_path": str(target).replace('\\', '/'), "source_url": url, "source_page": page,
            "download_status": "pending", "size_bytes": "",
        })
        urls.add(url)
    if not items:
        raise RuntimeError(f"No matching MAT downloads found at {page}; site structure may have changed")
    return items


def is_mat_file(path):
    if not path.is_file() or path.stat().st_size < 128:
        return False
    with path.open("rb") as f:
        prefix = f.read(128)
    # MAT v5 starts with a descriptive header; v7.3 is HDF5.
    return prefix.startswith(b"MATLAB") or prefix.startswith(b"\x89HDF\r\n\x1a\n")


def download(sess, item, root, delay):
    dest = root / item["local_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if is_mat_file(dest):
        item["download_status"] = "already_exists"
        item["size_bytes"] = dest.stat().st_size
        return
    if dest.exists():
        print(f"[WARN] Removing invalid/incomplete existing file: {dest}")
        dest.unlink()
    temp = dest.with_suffix(dest.suffix + ".part")
    try:
        with sess.get(item["source_url"], stream=True, timeout=(15, 120)) as res:
            res.raise_for_status()
            with temp.open("wb") as f:
                for chunk in res.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        if not is_mat_file(temp):
            raise ValueError("Downloaded content is not a valid MAT file header")
        temp.replace(dest)
        item["download_status"] = "downloaded"
        item["size_bytes"] = dest.stat().st_size
    except (requests.RequestException, OSError, ValueError) as e:
        item["download_status"] = f"ERROR: {type(e).__name__}: {str(e)[:160]}"
        if temp.exists():
            temp.unlink()
    time.sleep(delay)


def save_manifest(items, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(items)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="bearing-maintenance-ai/data/cwru", help="Download folder")
    parser.add_argument("--groups", nargs="+", choices=list(PAGES), default=list(PAGES),
                        help="Which sections to download")
    parser.add_argument("--dry-run", action="store_true", help="List downloads and write manifest, no MAT downloading")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between files (default 0.5)")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    sess = make_session()
    all_items = []
    for group in args.groups:
        try:
            items = discover(sess, group)
            print(f"[DISCOVERED] {group}: {len(items)} files")
            all_items.extend(items)
        except (requests.RequestException, RuntimeError) as e:
            print(f"[ERROR] Cannot discover {group}: {e}", file=sys.stderr)
    if not all_items:
        sys.exit("No files discovered. Check internet access and the CWRU website.")

    manifest = root / "metadata.csv"
    save_manifest(all_items, manifest)  # Also useful when a run is interrupted.
    if not args.dry_run:
        for index, item in enumerate(all_items, 1):
            print(f"[{index}/{len(all_items)}] {item['group']} / {item['site_label']}", end=" ", flush=True)
            download(sess, item, root, args.delay)
            print(item["download_status"])
            save_manifest(all_items, manifest)
    else:
        for item in all_items[:12]:
            print("  ", item["local_path"])
        if len(all_items) > 12:
            print(f"  ... and {len(all_items) - 12} more")
    print(f"\nTotal discovered: {len(all_items)}")
    print(f"Metadata written: {manifest.resolve()}")
    print("Run the command again to skip previously downloaded valid MAT files.")


if __name__ == "__main__":
    main()
