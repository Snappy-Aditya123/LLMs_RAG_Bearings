#!/usr/bin/env python3
"""Download official Paderborn Bearing DataCenter archives (all 32 states).

Install: pip install requests beautifulsoup4
Preview: python download_paderborn.py --dry-run
Download: python download_paderborn.py
Download and extract: python download_paderborn.py --extract
Resume simply by running the same command again.
"""

import argparse
import csv
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

INDEX = 'https://groups.uni-paderborn.de/kat/BearingDataCenter/'
FIELDS = ['bearing_id', 'condition_group', 'source_url', 'archive_path',
          'bytes_downloaded', 'status', 'extraction_path', 'extraction_status']


def session():
    s = requests.Session()
    s.headers.update({'User-Agent': 'Research-bearing-dataset-downloader/1.0'})
    retry = Retry(total=4, connect=4, read=4, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(['GET', 'HEAD']))
    s.mount('https://', HTTPAdapter(max_retries=retry))
    return s


def discover(s):
    response = s.get(INDEX, timeout=(15, 45))
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    result = {}
    for a in soup.find_all('a', href=True):
        href = unquote(a['href'])
        name = Path(urlparse(href).path).name
        if not re.fullmatch(r'(K\d{3}|KA\d{2}|KB\d{2}|KI\d{2})\.rar', name, re.I):
            continue
        url = urljoin(INDEX, a['href'])
        if urlparse(url).hostname != 'groups.uni-paderborn.de':
            continue
        ident = name[:-4].upper()
        group = 'healthy' if re.fullmatch(r'K\d{3}', ident) else 'damaged'
        # KA/KB/KI identifiers alone must not be interpreted as exact
        # damage mechanism or artificial-versus-real provenance.
        result[ident] = dict(bearing_id=ident, condition_group=group,
                             source_url=url,
                             archive_path=f'archives/{group}/{ident}.rar',
                             bytes_downloaded='', status='pending',
                             extraction_path=f'extracted/{group}/{ident}',
                             extraction_status='not_requested')
    if not result:
        raise RuntimeError('No RAR links discovered. The university index may have changed.')
    return [result[key] for key in sorted(result)]


def save_manifest(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def download(s, row, root, delay):
    target = root / row['archive_path']
    target.parent.mkdir(parents=True, exist_ok=True)
    part = Path(str(target) + '.part')
    if target.exists() and target.stat().st_size > 0:
        # Size-check a complete-looking local file against remote content length.
        try:
            r = s.head(row['source_url'], timeout=(15, 30), allow_redirects=True)
            expected = int(r.headers.get('Content-Length', '0')) if r.ok else 0
        except (requests.RequestException, ValueError):
            expected = 0
        if expected == 0 or target.stat().st_size == expected:
            row.update(status='already_exists', bytes_downloaded=target.stat().st_size)
            return
        print(f'  Existing size mismatch ({target.stat().st_size} vs {expected}); redownloading')
        target.unlink()
    try:
        start = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={start}-'} if start else {}
        with s.get(row['source_url'], headers=headers, stream=True,
                   timeout=(20, 180)) as r:
            # Restart if the server ignores Range, or the partial file cannot be
            # reconciled with the returned range. Do not append a full response.
            if r.status_code == 416 and start:
                part.unlink()
                return download(s, row, root, delay)
            r.raise_for_status()
            content_range = r.headers.get('Content-Range', '')
            continued = (start > 0 and r.status_code == 206 and
                         content_range.startswith(f'bytes {start}-'))
            mode = 'ab' if continued else 'wb'
            if start and not continued:
                print('  Server did not honor resume; restarting this archive')
            with part.open(mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            total = part.stat().st_size
            if continued and '/' in content_range:
                expected_total = content_range.rsplit('/', 1)[-1]
                if expected_total.isdigit() and total != int(expected_total):
                    raise IOError(f'Partial download: {total}/{expected_total} bytes')
            elif 'Content-Length' in r.headers:
                expected = int(r.headers['Content-Length']) + (start if continued else 0)
                if expected != total:
                    raise IOError(f'Incomplete download: {total}/{expected} bytes')
        if total < 1024:
            raise IOError('Unexpectedly small download; possible error page')
        # RAR4 / RAR5 magic headers
        with part.open('rb') as f:
            header = f.read(8)
        if not (header.startswith(b'Rar!\x1a\x07\x00') or header.startswith(b'Rar!\x1a\x07\x01\x00')):
            raise IOError('Downloaded file does not have a RAR archive header')
        part.replace(target)
        row.update(status='downloaded', bytes_downloaded=total)
    except (OSError, requests.RequestException, ValueError) as e:
        row.update(status=f'ERROR: {e}', bytes_downloaded=part.stat().st_size if part.exists() else 0)
    time.sleep(delay)


def extract(row, root):
    source = root / row['archive_path']
    if not source.is_file() or row['status'].startswith('ERROR'):
        row['extraction_status'] = 'skipped_missing_archive'
        return
    dst = root / row['extraction_path']
    sentinel = dst / '.extraction_complete'
    if sentinel.exists():
        row['extraction_status'] = 'already_extracted'
        return
    tool = shutil.which('7z') or shutil.which('7zz') or shutil.which('7za')
    if tool is None:
        row['extraction_status'] = 'missing_7zip (install 7-Zip and add 7z to PATH)'
        return
    dst.mkdir(parents=True, exist_ok=True)
    try:
        # The archives come from the university. Extract into an isolated directory.
        subprocess.run([tool, 'x', '-y', f'-o{dst}', str(source.resolve())],
                       check=True, stdout=subprocess.DEVNULL)
        sentinel.write_text('Completed successfully\n', encoding='utf-8')
        row['extraction_status'] = 'extracted'
    except (OSError, subprocess.CalledProcessError) as e:
        row['extraction_status'] = f'ERROR: {e}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='bearing-maintenance-ai/data/paderborn')
    p.add_argument('--dry-run', action='store_true', help='Discover files without downloading')
    p.add_argument('--extract', action='store_true', help='Also extract using installed 7z/7zz')
    p.add_argument('--delay', type=float, default=1.0, help='Delay between files in seconds')
    p.add_argument('--only', nargs='+', metavar='BEARING_ID', help='Optional subset, e.g. K001 KA01 KI01')
    args = p.parse_args()
    root = Path(args.output)
    s = session()
    try:
        rows = discover(s)
    except (requests.RequestException, RuntimeError) as e:
        sys.exit(f'Failed to discover dataset links: {e}')
    if args.only:
        requested = {x.upper() for x in args.only}
        rows = [r for r in rows if r['bearing_id'] in requested]
        missing = requested - {r['bearing_id'] for r in rows}
        if missing:
            print(f'Warning: unknown bearing IDs: {sorted(missing)}', file=sys.stderr)
    if not rows:
        sys.exit('No matching bearings selected.')
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.csv'
    save_manifest(rows, manifest)
    print(f'Discovered {len(rows)} bearing archives from {INDEX}')
    if args.dry_run:
        for row in rows:
            print(row['bearing_id'], row['condition_group'], row['source_url'])
    else:
        for n, row in enumerate(rows, 1):
            print(f'[{n}/{len(rows)}] {row["bearing_id"]}', flush=True)
            download(s, row, root, max(args.delay, 0))
            if args.extract:
                extract(row, root)
            print(f'  {row["status"]} / {row["extraction_status"]}', flush=True)
            save_manifest(rows, manifest)
    print(f'Manifest: {manifest.resolve()}')
    print('Re-run this command to skip finished archives and resume partial downloads.')


if __name__ == '__main__':
    main()
