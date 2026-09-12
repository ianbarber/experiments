"""CPU-only checks for one explicit permission-error recovery; no model changes."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ORIGINAL_FREEZE_SHA256 = '0231a51e7efd3f0f166c0f347b4c82dfc3dc6edc008997e09ec2ce8a503f066f'
REUSABLE = ('s1729_competence_epoch1', 's1729_competence_select_epoch1')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify_original(root=ROOT):
    root = Path(root)
    if sha(root / 'FREEZE.json') != ORIGINAL_FREEZE_SHA256:
        raise ValueError('Original experiment freeze identity differs.')
    frozen = read(root / 'FREEZE.json')
    for name, digest in frozen['files'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Original frozen file changed: {name}')
    return read(root / 'configs/pilot.json')


def relative(root, path):
    return str(Path(path).resolve().relative_to(Path(root).resolve()))


def container_path(root, path):
    return '/workspace/' + relative(root, path) if path is not None else None


def expected_args(root, config, name, mode, data, seed, adapter=None, epoch=1, optimizer=None, sample=False):
    return dict(mode=mode, data=container_path(root, data),
                output='/workspace/results/stages/' + name,
                adapter=container_path(root, adapter), optimizer=container_path(root, optimizer),
                seed=seed, batch_size=config['micro_batch_size'] if mode == 'train' else config['generation_batch_size'],
                effective_batch=config['effective_batch'] if mode == 'train' else 16,
                epoch=epoch if mode == 'train' else 1,
                lr=config['learning_rate'] if mode == 'train' else 1e-4,
                sample=bool(sample), max_new_tokens=192, check_only=False)


def verify_stage(root, config, name, mode, data, seed, adapter=None, epoch=1, optimizer=None, sample=False,
                 expected_completion_sha256=None):
    root = Path(root)
    directory = root / 'results/stages' / name
    path = directory / 'COMPLETED.json'
    if not path.is_file():
        raise ValueError(f'Existing stage is incomplete; retry forbidden: {name}')
    if (directory / 'FAILED.json').exists():
        raise ValueError(f'Stage has a failure marker; reuse forbidden: {name}')
    if expected_completion_sha256 is not None and sha(path) != expected_completion_sha256:
        raise ValueError(f'Bound completion manifest changed: {name}')
    completed = read(path)
    if completed.get('status') != 'complete' or completed.get('mode') != mode or completed.get('check_only') is not False:
        raise ValueError(f'Stage is not a completed actual {mode}: {name}')
    wanted = expected_args(root, config, name, mode, data, seed, adapter, epoch, optimizer, sample)
    if completed['args'] != wanted:
        raise ValueError(f'Exact stage arguments differ: {name}')
    if completed['config'] != config or completed['config_sha256'] != sha(root / 'configs/pilot.json'):
        raise ValueError(f'Stage model/protocol configuration changed: {name}')
    sources = {relative(root, source): sha(source) for source in sorted((root / 'scripts').glob('*.py'))}
    if completed['source_sha256'] != sources:
        raise ValueError(f'Stage source inventory or hashes differ: {name}')
    if completed['data_sha256'] != sha(data):
        raise ValueError(f'Stage data changed: {name}')
    identity = {filename: sha(Path(adapter) / filename) for filename in ('adapter_model.safetensors', 'adapter_config.json')} if adapter else None
    if completed['initial_adapter_identity'] != identity:
        raise ValueError(f'Input adapter weights/configuration changed: {name}')
    if completed['initial_optimizer_sha256'] != (sha(optimizer) if optimizer else None):
        raise ValueError(f'Input optimizer changed: {name}')
    artifacts = completed['artifacts_sha256']
    required = {'started.json', 'tokenization.jsonl'} | (
        {'training.jsonl', 'optimizer.pt', 'adapter/adapter_model.safetensors', 'adapter/adapter_config.json', 'adapter/manifest.json'}
        if mode == 'train' else {'outputs.jsonl'})
    if not required.issubset(artifacts):
        raise ValueError(f'Stage completion omits required artifacts: {name}')
    for name_in_stage, digest in artifacts.items():
        candidate = (directory / name_in_stage).resolve()
        candidate.relative_to(directory.resolve())
        if sha(candidate) != digest:
            raise ValueError(f'Completed stage artifact changed: {name}/{name_in_stage}')
    if mode == 'generate':
        settings = dict(do_sample=bool(sample), temperature=0.7 if sample else 1.0,
                        top_p=0.95 if sample else 1.0, top_k=0, repetition_penalty=1.0,
                        max_new_tokens=192, num_beams=1, num_return_sequences=1, min_new_tokens=0,
                        pad_token_id=151643, eos_token_id=151645, bos_token_id=None,
                        use_cache=True, return_dict_in_generate=False)
        if completed['generation_settings'] != settings:
            raise ValueError(f'Explicit generation settings differ: {name}')
    return {'completion_sha256': sha(path), 'mode': mode, 'args': wanted,
            'artifacts_sha256': artifacts, 'data_sha256': completed['data_sha256']}


def known_calls(root=ROOT):
    root = Path(root)
    return [dict(name=REUSABLE[0], mode='train', data=root / 'data/competence_train.jsonl', seed=1729),
            dict(name=REUSABLE[1], mode='generate', data=root / 'data/competence_select.jsonl', seed=1729,
                 adapter=root / 'results/stages' / REUSABLE[0] / 'adapter')]


def prior_budget(root, prior_directory, config):
    root, prior_directory = Path(root), Path(prior_directory)
    allocation = read(prior_directory / 'allocation.json')
    released = read(prior_directory / 'allocation_released.json')
    terminal = read(prior_directory / 'TERMINAL.json')
    restored = read(prior_directory / 'service_restoration.json')
    if (terminal.get('status') != 'operational_error' or terminal.get('repair_training_launched') is not False
            or terminal.get('freeze_sha256') != ORIGINAL_FREEZE_SHA256
            or terminal.get('failure', {}).get('error_type') != 'PermissionError'
            or 'recomputed_scores.json.tmp' not in terminal.get('failure', {}).get('message', '')):
        raise ValueError('Recovery is restricted to the recorded host-side score-file PermissionError.')
    if (released.get('research_running') is not False or restored.get('health_status') != 200
            or restored.get('same_original_container_and_image') is not True or restored.get('research_running') is not False):
        raise ValueError('Previous GPU allocation/service restoration is not verified released and healthy.')
    if (restored['service_id'] != allocation['service_id'] or restored['service_image'] != allocation['service_image']
            or allocation['research_image'] != config['container_image']):
        raise ValueError('Prior allocation/restoration identities disagree.')
    total = config['gpu_budget_seconds']
    if total != 28800 or allocation['budget_seconds'] != total:
        raise ValueError('Unexpected original allocation budget; no implicit additional allocation.')
    elapsed = released['released_unix'] - allocation['started_unix']
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Invalid original allocation timestamps.')
    charged = math.ceil(elapsed)
    remaining = total - charged
    if remaining <= 180:
        raise ValueError('Original eight-hour budget is exhausted or only the shutdown reserve remains.')
    return dict(original_budget_seconds=total, prior_elapsed_seconds=elapsed,
                prior_charged_seconds=charged, remaining_budget_seconds=remaining,
                shutdown_reserve_seconds=180, original_service_id=allocation['service_id'],
                original_service_image=allocation['service_image'])


def make_plan(prior_directory, root=ROOT, here=HERE):
    root, here, prior_directory = Path(root), Path(here), Path(prior_directory)
    config = verify_original(root)
    budget = prior_budget(root, prior_directory, config)
    stage_directories = {path.name for path in (root / 'results/stages').iterdir() if path.is_dir()}
    if stage_directories != set(REUSABLE):
        raise ValueError('Recovery expects exactly the two recorded completed stages; additional/incomplete stages require separate review.')
    stages = {call['name']: verify_stage(root, config, **call) for call in known_calls(root)}
    code_files = sorted(here.glob('*.py'))
    if not {'resume_common.py', 'resume_program.py', 'resume_session.py', 'test_resume.py'}.issubset({path.name for path in code_files}):
        raise ValueError('Recovery implementation/test inventory is incomplete.')
    prior_files = sorted(path for path in prior_directory.rglob('*') if path.is_file())
    return dict(scope='Single explicit operational recovery; unchanged preliminary experiment',
                original_freeze_sha256=ORIGINAL_FREEZE_SHA256,
                recovery_code_sha256={relative(root, path): sha(path) for path in code_files},
                prior_attempt_directory=relative(root, prior_directory),
                prior_artifacts_sha256={relative(root, path): sha(path) for path in prior_files},
                reusable_stages=stages, budget=budget, repair_training_authorized=False,
                fresh_qualification_retry_authorized=False, incomplete_stage_retry_authorized=False)


def verify_plan(root=ROOT, here=HERE):
    root, here = Path(root), Path(here)
    plan = read(here / 'RESUME_FREEZE.json')
    config = verify_original(root)
    if plan['original_freeze_sha256'] != ORIGINAL_FREEZE_SHA256 or set(plan['reusable_stages']) != set(REUSABLE):
        raise ValueError('Recovery freeze scope differs from the reviewed two-stage continuation.')
    if any(plan.get(key) is not False for key in ('repair_training_authorized', 'fresh_qualification_retry_authorized', 'incomplete_stage_retry_authorized')):
        raise ValueError('Recovery freeze improperly authorizes new scientific work or retries.')
    current_code = {relative(root, path): sha(path) for path in sorted(here.glob('*.py'))}
    if current_code != plan['recovery_code_sha256']:
        raise ValueError('Recovery source/test inventory changed after its separate freeze.')
    for name, digest in plan['prior_artifacts_sha256'].items():
        if sha(root / name) != digest:
            raise ValueError(f'Archived operational record changed: {name}')
    budget = prior_budget(root, root / plan['prior_attempt_directory'], config)
    if budget != plan['budget']:
        raise ValueError('Remaining-budget calculation changed after recovery freeze.')
    for call in known_calls(root):
        verify_stage(root, config, **call,
                     expected_completion_sha256=plan['reusable_stages'][call['name']]['completion_sha256'])
    return plan, config
