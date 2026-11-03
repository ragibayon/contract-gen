"""Restore the byte-for-byte saved results; refuse to overwrite different files."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import tempfile

archive_dir = Path(__file__).resolve().parent
destination = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else archive_dir.parent
manifest = json.loads((archive_dir / 'manifest.json').read_text())
with tarfile.open(archive_dir / manifest['archive'], 'r:gz') as archive:
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe manifest path')
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as output:
            temporary = Path(output.name)
            try:
                source = archive.extractfile(entry['path'])
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    digest.update(block)
                    output.write(block)
                output.flush()
                if digest.hexdigest() != entry['sha256'] or temporary.stat().st_size != entry['size']:
                    raise ValueError(f'Checksum mismatch: {relative}')
                if target.exists():
                    existing = hashlib.sha256()
                    with target.open('rb') as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b''):
                            existing.update(block)
                    if existing.hexdigest() != entry['sha256']:
                        raise FileExistsError(f'Refusing to overwrite: {target}')
                else:
                    temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
print(f"Verified and restored {len(manifest['files'])} files under {destination}")
