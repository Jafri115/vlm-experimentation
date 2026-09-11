"""Download pinned official Qwen GGUF weights and a portable llama.cpp runtime."""
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = '7c41481f57cb95916b40956ab2f0b139b296d974'
RELEASE = 'b10909'


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download(url, path, expected_hash):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == expected_hash:
        print(f'Already verified: {path.name}', flush=True)
        return
    partial = path.with_suffix(path.suffix+'.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
    with urllib.request.urlopen(request, timeout=120) as response:
        resume = offset and response.status == 206
        total = int(response.headers.get('Content-Length', 0)) + (offset if resume else 0)
        count, last = offset if resume else 0, time.monotonic()
        with partial.open('ab' if resume else 'wb') as stream:
            while chunk := response.read(8*1024*1024):
                stream.write(chunk)
                count += len(chunk)
                if time.monotonic()-last > 15:
                    print(f'{path.name}: {count/2**20:.0f}/{total/2**20:.0f} MiB', flush=True)
                    last = time.monotonic()
    if digest(partial) != expected_hash:
        raise ValueError(f'Checksum mismatch: {partial}')
    partial.replace(path)
    print(f'Verified: {path.name}', flush=True)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    model_dir = ROOT/'models/qwen3_8b'
    runtime_dir = ROOT/'models/llama_cpp_b10909_vulkan'
    model_name = 'Qwen3-8B-Q4_K_M.gguf'
    info = get_json(f'https://huggingface.co/api/models/Qwen/Qwen3-8B-GGUF/revision/{REVISION}?blobs=true')
    entry = next(r for r in info['siblings'] if r['rfilename'] == model_name)
    expected = entry['lfs']['sha256']
    release = get_json(f'https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/{RELEASE}')
    asset = next(a for a in release['assets'] if a['name'] == f'llama-{RELEASE}-bin-win-vulkan-x64.zip')
    asset_hash = asset['digest'].removeprefix('sha256:')
    archive = runtime_dir/asset['name']
    download(asset['browser_download_url'], archive, asset_hash)
    with zipfile.ZipFile(archive) as package:
        for member in package.infolist():
            resolved = (runtime_dir/member.filename).resolve()
            if not resolved.is_relative_to(runtime_dir.resolve()):
                raise ValueError('Unsafe archive path')
        package.extractall(runtime_dir)
    download(f'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/{REVISION}/{model_name}', model_dir/model_name, expected)
    metadata = {'model_id': 'Qwen/Qwen3-8B', 'gguf_repo': 'Qwen/Qwen3-8B-GGUF', 'revision': REVISION,
                'quantization': 'Q4_K_M', 'model_path': str(model_dir/model_name), 'model_sha256': expected,
                'llama_cpp_release': RELEASE, 'runtime_archive_sha256': asset_hash,
                'server_path': str(next(runtime_dir.rglob('llama-server.exe')))}
    (model_dir/'provenance.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == '__main__':
    main()
