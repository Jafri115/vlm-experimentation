"""Download the Cohere dataset release with credentials held only in memory.

Run interactively: both AK and SK are prompted without echo. Alternatively read
OTC_OBS_AK/OTC_OBS_SK from the current process environment. No keys are persisted.
"""
from __future__ import annotations

import argparse
import base64
import email.utils
import getpass
import hashlib
import hmac
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

DEFAULT_URI = 'obs://protect-ai/transcription_artifacts/datasets/v1/memopsy_196_dataset_versions_v1.tar.gz'
DEFAULT_ENDPOINT = 'https://obs.eu-de.otc.t-systems.com'


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError('OBS redirected the request. Check the endpoint; credentials were not forwarded.')


def authorization(ak, sk, bucket, key, date, scheme):
    canonical = f'GET\n\n\n{date}\n/{bucket}/{key}'
    signature = base64.b64encode(hmac.new(sk.encode(), canonical.encode(), hashlib.sha1).digest()).decode()
    return f'{scheme} {ak}:{signature}'


def credential(name, prompt):
    value = os.environ.get(name, '').strip()
    if not value:
        # Refuse getpass's fallback to visible input when no terminal is available.
        with warnings.catch_warnings():
            warnings.simplefilter('error', getpass.GetPassWarning)
            value = getpass.getpass(prompt).strip()
    if not value:
        raise ValueError('An AK and SK are both required')
    return value


def download(args):
    uri = urllib.parse.urlsplit(args.uri)
    endpoint = urllib.parse.urlsplit(args.endpoint)
    if uri.scheme != 'obs' or not uri.netloc or not uri.path or uri.query or uri.fragment:
        raise ValueError('Expected obs://bucket/object without query parameters')
    if endpoint.scheme != 'https' or not endpoint.hostname or endpoint.path not in ('', '/') or endpoint.query:
        raise ValueError('Endpoint must be an HTTPS service hostname')
    if endpoint.username or endpoint.password or endpoint.port:
        raise ValueError('Use the service endpoint without embedded credentials or port')
    bucket = uri.netloc
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789.-' for c in bucket):
        raise ValueError('Invalid bucket name')
    key = urllib.parse.unquote(uri.path.lstrip('/'))
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f'File already exists; inspect it or choose a fresh output: {output}')
    partial = output.with_name(output.name + '.part')
    if partial.exists():
        raise ValueError(f'Partial file already exists; choose a fresh output or remove that partial: {partial}')
    ak = credential('OTC_OBS_AK', 'OBS Access Key (AK; hidden): ')
    sk = credential('OTC_OBS_SK', 'OBS Secret Key (SK; hidden): ')
    date = email.utils.formatdate(usegmt=True)
    url = f'https://{bucket}.{endpoint.hostname}/{urllib.parse.quote(key, safe="/~")}'
    req = urllib.request.Request(url, headers={
        'Date': date, 'Authorization': authorization(ak, sk, bucket, key, date, args.auth_scheme)})
    del ak, sk
    opener = urllib.request.build_opener(NoRedirect())
    output.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    received = 0
    next_progress = 32 * 1024 * 1024
    try:
        with opener.open(req, timeout=90) as response, partial.open('xb') as handle:
            expected = response.headers.get('Content-Length')
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                hasher.update(block)
                received += len(block)
                if received >= next_progress:
                    print(f'Downloaded {received / 1024**2:.1f} MiB', flush=True)
                    next_progress = received + 32 * 1024 * 1024
            if expected is not None and received != int(expected):
                raise RuntimeError('Incomplete download; kept .part file, archive was not promoted')
        if not received:
            raise RuntimeError('Received an empty object')
        partial.rename(output)
    except urllib.error.HTTPError as exc:
        # Never print the signed request, error body or headers.
        raise RuntimeError(f'OBS returned HTTP {exc.code}. Check AK/SK, bucket read permission and endpoint. '
                           'If this is an S3-compatible bucket, retry with --auth-scheme AWS.') from None
    print(f'Download complete: {output}')
    print(f'Bytes: {received}; SHA256: {hasher.hexdigest()}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--uri', default=DEFAULT_URI)
    p.add_argument('--endpoint', default=DEFAULT_ENDPOINT)
    p.add_argument('--auth-scheme', choices=['OBS', 'AWS'], default='OBS')
    p.add_argument('--output', type=Path,
                   default=Path('artifacts/cohere_dataset_release/memopsy_196_dataset_versions_v1.tar.gz'))
    try:
        download(p.parse_args())
    except (ValueError, RuntimeError, OSError, getpass.GetPassWarning) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
