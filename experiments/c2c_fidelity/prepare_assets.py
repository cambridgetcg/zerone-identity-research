"""Fetch only the pinned public C2C sources/assets listed in our manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, required=True)
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[2]
    destination = args.workdir.expanduser().resolve()
    if destination == repository or destination.is_relative_to(repository):
        raise SystemExit('Keep large assets outside the research repository.')
    manifest = json.loads((repository/'evidence/c2c-fidelity-sources.json').read_text())
    destination.mkdir(parents=True, exist_ok=True)
    upstream = destination/'upstream'
    source = manifest['upstream']
    if not upstream.exists():
        subprocess.run(['git','init',str(upstream)], check=True)
        subprocess.run(['git','-C',str(upstream),'remote','add','origin',source['repository']], check=True)
        subprocess.run(['git','-C',str(upstream),'fetch','--depth','1','origin',source['revision']], check=True)
        subprocess.run(['git','-C',str(upstream),'checkout','--detach',source['revision']], check=True)
    observed = subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'], text=True).strip()
    if observed != source['revision']:
        raise SystemExit('Existing upstream checkout has a different revision; use a separate workdir.')
    for item in source['files']:
        if sha256(upstream/item['path']) != item['sha256']:
            raise SystemExit('Upstream source digest mismatch: '+item['path'])
    from huggingface_hub import snapshot_download
    for asset in manifest['assets']:
        folder = destination/'assets'/asset['alias']
        snapshot_download(repo_id=asset['repo'], revision=asset['revision'],
                          allow_patterns=[item['path'] for item in asset['files']],
                          local_dir=str(folder), token=False, max_workers=3)
        for item in asset['files']:
            path = folder/item['path']
            if path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise SystemExit('Public asset digest mismatch: '+item['path'])
        print('Verified', asset['repo'], asset['revision'], flush=True)
    print('Pinned public assets are ready. Inference can run offline.')


if __name__ == '__main__':
    main()
