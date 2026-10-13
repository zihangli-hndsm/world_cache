"""Stage portable evidence copies, preserving original files and provenance hashes."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
from generate_submission_manifest import manifest_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output == root or (output.exists() and any(output.iterdir())):
        raise ValueError('output must be a fresh staging directory')
    entries = []
    for relative in manifest_files(root):
        source = root / relative
        original = source.read_bytes()
        content = original
        if source.suffix in {'.json', '.csv', '.md', '.txt', '.py', '.sh', '.tex', '.yaml', '.yml'}:
            # Replace only machine path prefixes; preserve scientific numbers and fields.
            content = re.sub(rb'/home/[A-Za-z0-9._-]+/world_cache/', b'./', content)
            content = re.sub(rb'/home/[A-Za-z0-9._-]+/', b'${HOME}/', content)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append({'path': relative,
                        'original_sha256': hashlib.sha256(original).hexdigest(),
                        'portable_sha256': hashlib.sha256(content).hexdigest(),
                        'path_prefixes_replaced': content != original})
    (output / 'PORTABLE_COPY_PROVENANCE.json').write_text(json.dumps({
        'policy': 'Only machine-local path prefixes replaced; originals retained. ${HOME} paths in evidence are descriptive placeholders.',
        'files': entries}, indent=2) + '\n')
    print(f'Staged {len(entries)} files; path prefixes changed in {sum(e["path_prefixes_replaced"] for e in entries)} files')


if __name__ == '__main__':
    main()
