"""Validate recorded checkpoint-branch artifacts without neural inference."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check():
    receipt = json.loads((ROOT / 'evidence/checkpoint-branches-receipt.json').read_text())
    assert receipt['schema'] == 'checkpoint-branches-receipt-v1'
    assert re.fullmatch(r'[0-9a-f]{40}', receipt['protocol_commit'])
    assert receipt['new_model_downloads'] is False
    assert receipt['training_performed'] is False
    assert receipt['private_state_exported'] is False
    expected = {
        'experiments/checkpoint_branches/run_experiment.py',
        'experiments/checkpoint_branches/analyze_results.py',
        'experiments/checkpoint_branches/fixtures.json',
        'experiments/checkpoint_branches/PROTOCOL.md',
        'experiments/c2c_fidelity/checkpoint_control.py',
        'experiments/c2c_fidelity/requirements-lock.txt',
        'evidence/c2c-fidelity-sources.json',
        'evidence/checkpoint-branches-results.json',
        'evidence/checkpoint-branches-summary.json',
        'evidence/checkpoint-branches-run.log',
    }
    assert set(receipt['artifact_sha256']) == expected
    for relative, sha in receipt['artifact_sha256'].items():
        assert re.fullmatch(r'[0-9a-f]{64}', sha)
        assert digest(ROOT / relative) == sha
    result = json.loads((ROOT / 'evidence/checkpoint-branches-results.json').read_text())
    source = result['provenance']
    for field, path in {
        'harness_sha256': 'experiments/checkpoint_branches/run_experiment.py',
        'helper_sha256': 'experiments/c2c_fidelity/checkpoint_control.py',
        'fixtures_sha256': 'experiments/checkpoint_branches/fixtures.json',
        'asset_manifest_sha256': 'evidence/c2c-fidelity-sources.json',
    }.items():
        assert source[field] == receipt['artifact_sha256'][path]
    manifest = json.loads((ROOT / 'evidence/c2c-fidelity-sources.json').read_text())
    receiver = next(asset for asset in manifest['assets'] if asset['alias'] == 'receiver')
    assert source['model'] == receiver['repo'] and source['model_revision'] == receiver['revision']
    assert source['model_files'] == receiver['files']
    subprocess.run([
        sys.executable, 'experiments/checkpoint_branches/analyze_results.py',
        '--input', 'evidence/checkpoint-branches-results.json',
        '--fixtures', 'experiments/checkpoint_branches/fixtures.json',
        '--output', 'evidence/checkpoint-branches-summary.json', '--check-existing',
    ], cwd=ROOT, check=True, timeout=30)
    print(json.dumps({'checkpoint_branch_artifacts_checked': True,
                      'neural_inference_rerun': False,
                      'remote_model_bytes_rechecked': False}))


if __name__ == '__main__':
    check()
