"""Check recorded C2C provenance and arithmetic without running neural inference."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check():
    manifest = read('evidence/c2c-fidelity-sources.json')
    receipt = read('evidence/c2c-run-receipt.json')
    source = manifest['upstream']
    assert re.fullmatch(r'[0-9a-f]{40}', source['revision'])
    source_hashes = {item['path']: item['sha256'] for item in source['files']}
    assert len(source_hashes) == len(source['files'])
    for item in source['files']:
        assert item['url'] == ('https://github.com/thu-nics/C2C/blob/'
                               + source['revision'] + '/' + item['path'])
    assert {asset['alias'] for asset in manifest['assets']} == {'receiver', 'donor', 'fuser'}
    for asset in manifest['assets']:
        assert re.fullmatch(r'[0-9a-f]{40}', asset['revision'])
        assert asset['url'] == 'https://huggingface.co/' + asset['repo'] + '/tree/' + asset['revision']
    for group in [source, *manifest['assets']]:
        assert len({item['path'] for item in group['files']}) == len(group['files'])
        for item in group['files']:
            assert re.fullmatch(r'[0-9a-f]{64}', item['sha256'])
            assert not Path(item['path']).is_absolute() and '..' not in Path(item['path']).parts
            assert type(item['bytes']) is int and item['bytes'] > 0
    fuser = next(asset for asset in manifest['assets'] if asset['alias'] == 'fuser')
    fuser_hashes = {Path(item['path']).name: item['sha256'] for item in fuser['files']}
    for prefix, fixture, receipt_key in [
        ('c2c-fidelity', 'fixtures.json', 'primary_result_sha256'),
        ('c2c-answer-prefix', 'fixtures-answer-prefix.json', 'followup_result_sha256'),
    ]:
        result_path = ROOT / f'evidence/{prefix}-results.json'
        result = json.loads(result_path.read_text())
        assert digest(result_path) == receipt[receipt_key]
        assert result['upstream_revision'] == source['revision']
        assert set(result['source_sha256']) == {
            'rosetta/model/wrapper.py', 'rosetta/model/projector.py',
            'rosetta/model/sampling.py', 'rosetta/utils/registry.py',
        }
        assert all(source_hashes[path] == sha for path, sha in result['source_sha256'].items())
        expected_projectors = {path for path in fuser_hashes
                               if re.fullmatch(r'projector_[0-9]+\.(json|pt)', path)}
        assert len(expected_projectors) == 56
        assert set(result['checkpoint']['checkpoint_file_sha256']) == expected_projectors
        assert all(fuser_hashes[path] == sha
                   for path, sha in result['checkpoint']['checkpoint_file_sha256'].items())
        subprocess.run([
            sys.executable, 'experiments/c2c_fidelity/analyze_results.py',
            '--input', f'evidence/{prefix}-results.json',
            '--fixtures', f'experiments/c2c_fidelity/{fixture}',
            '--output', f'evidence/{prefix}-summary.json', '--check-existing',
        ], cwd=ROOT, check=True, timeout=30)
    checkpoint = read('evidence/c2c-native-checkpoint-result.json')
    assert checkpoint['passed'] and checkpoint['all_reloaded_tensors_bitwise_equal']
    assert checkpoint['same_chunking_uninterrupted_vs_restored']['maximum_absolute_logit_difference'] == 0
    assert checkpoint['greedy_continuation_ids'][0] == checkpoint['greedy_continuation_ids'][1]
    assert len(checkpoint['continuation_comparisons']) == 4
    for comparison in checkpoint['continuation_comparisons']:
        assert comparison['maximum_absolute_logit_difference'] == comparison['total_variation_distance'] == 0
        assert comparison['greedy_token_equal']
    receiver = next(asset for asset in manifest['assets'] if asset['alias'] == 'receiver')
    assert checkpoint['model'] == receiver['repo'] and checkpoint['model_revision'] == receiver['revision']
    for name, sha in receipt['final_experiment_file_sha256'].items():
        assert digest(ROOT / 'experiments/c2c_fidelity' / name) == sha
    assert digest(ROOT / 'evidence/c2c-native-checkpoint-result.json') == receipt['checkpoint_result_sha256']
    print(json.dumps({'c2c_recorded_runs_validated': 2, 'native_checkpoint_record_validated': True,
                      'neural_inference_rerun': False, 'remote_asset_bytes_rechecked': False}))


if __name__ == '__main__':
    check()
