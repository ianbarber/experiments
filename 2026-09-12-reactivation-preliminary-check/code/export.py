"""Build a NEW public export draft only after the GPU session has ended.

Writes no source experiment files and never overwrites a destination. This is a
publication projection, not an original-byte copy of omitted private artifacts.
The three top-level editorial documents are drafts requiring lead review.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
from functools import lru_cache
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
SLUG = '2026-09-12-reactivation-preliminary-check'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compressed_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = ''.join(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n' for row in rows).encode()
    path.write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))


def operational_replacements(source):
    root = Path(source)
    result = {}
    for name in ('results/operational_resume/prior_attempt/allocation.json',
                 'results/operational_resume/CONTAINER_FIX.json', 'results/allocation.json'):
        path = root / name
        if path.exists():
            result.update(cached_operational_replacements(str(path), path.stat().st_mtime_ns))
    return result


@lru_cache(maxsize=8)
def cached_operational_replacements(path, modified_ns):
    state = read_json(path)
    return {state[key]: replacement for key, replacement in (('service_id', 'original-serving-container'), ('research_id', 'isolated-research-container'),
            ('prior_container_id', 'prior-research-container'), ('new_container_id', 'isolated-research-container')) if state.get(key)}


def safe_text(text, source, normalize_container_paths=True):
    for original, replacement in operational_replacements(str(source)).items():
        text = text.replace(original, replacement)
    text = text.replace(str(source.resolve()) + '/', '').replace(str(source.resolve()), 'experiment-workspace')
    if normalize_container_paths:
        text = text.replace('/workspace/', '').replace('/workspace', 'experiment-workspace')
    # No actual home pathname is needed in a public record. Do not publish the
    # private username as a replacement-map key either.
    text = re.sub(r'/home/[^/\s"\x27]+(?:/[^\s"\x27<>)]*)?', '[private-path-removed]', text)
    return text


PRIVATE_KEYS = {'service_id', 'research_id', 'service_image', 'research_image', 'hostname', 'host', 'pid', 'session_pid',
                'container_id', 'prior_container_id', 'new_container_id', 'original_service_id', 'original_service_image',
                'token_ids', 'input_ids', 'labels', 'optimizer_state_dict'}


def project(value, source):
    if isinstance(value, dict):
        if 'text' in value and 'finish_reason' in value and safe_text(value['text'], source) != value['text']:
            raise ValueError('Privacy redaction would alter generated scientific text; explicit publication review is required')
        for key in ('before', 'after', 'text', 'generated_text'):
            text = value.get(key)
            if isinstance(text, str) and '<decision>' in text and safe_text(text, source) != text:
                raise ValueError('Privacy redaction would alter generated scientific text; explicit publication review is required')
        output = {}
        for key, item in value.items():
            if key in PRIVATE_KEYS:
                continue
            if key == 'unexpected_special_token_ids':
                output['unexpected_special_token'] = bool(item)
            elif key == 'config':
                # Frozen public config exists separately; the source hash stays.
                continue
            elif key == 'trainable_parameter_names':
                output['trainable_parameter_tensor_count'] = len(item)
            else:
                output[safe_text(key, source)] = project(item, source)
        return output
    if isinstance(value, list):
        return [project(item, source) for item in value]
    if isinstance(value, str):
        return safe_text(value, source)
    return value


def source_record(source, path, destination, public_path, mode, original_sha=None):
    return dict(original_path=str(path.relative_to(source)), original_sha256=original_sha or sha(path),
                public_path=public_path, public_sha256=sha(destination / public_path), mode=mode)


def export_historical_cohort_review(source, destination, transport):
    transport['historical_reference_projections'] = []
    for filename in ('ACTUAL_CONTENT_QUALITY_REVIEW.md', 'PLAN.md', 'DESIGN_DETAILS.md', 'CRITICAL_REVIEW.md'):
        companion = source / 'references' / filename
        if not companion.exists():
            continue
        public = 'code/references/' + filename
        target = destination / public
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(safe_text(companion.read_text(), source))
        record = source_record(source, companion, destination, public, 'historical_reference_snapshot')
        transport['historical_reference_projections'].append(record)
        transport['projected_records'].append(record)
    name = 'references/FINAL_COHORT_REVIEW.md'
    path = source / name
    if not path.exists():
        return
    targets = ['../results/cohort_diagnostics/final/' + filename for filename in
               ('summary.json', 'realized_pair_review_manifest.json', 'realized_pair_review.jsonl')]
    text = safe_text(path.read_text(), source)
    for target in targets:
        pattern = re.compile(r'\[([^\]]+)\]\(' + re.escape(target) + r'\)')
        text, count = pattern.subn(lambda match: match.group(1) + ' (original local path: `' + target + '`)', text)
        if count != 1:
            raise ValueError('Historical cohort diagnostic-link inventory differs from the reviewed three links')
    text += ('\n## Archive note\n\nThese original local diagnostics are not republished here. '
             'The [published pilot evidence](../../../2026-09-09-reactivation-repair-pilot/README.md) is available in the earlier entry; '
             'the paths and hashes above identify the original diagnostics.\n')
    public = 'code/' + name
    target = destination / public
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)
    record = source_record(source, path, destination, public, 'historical_link_projection')
    record['projection'] = 'Three unpublished local diagnostic links rendered as inline paths; original labels and hashes preserved; published-pilot archive note appended without claiming byte identity.'
    transport['historical_reference_projections'].append(record)
    transport['projected_records'].append(record)
    if name in transport['frozen_sources']:
        transport['frozen_sources'][name].update(mode='historical_link_projection', public_path=public, public_sha256=sha(target))


def audit_resume_source(source):
    """Independently bind the explicit recovery, without importing its helpers."""
    source = Path(source)
    path = source / 'results/operational_resume/RESUME_FREEZE.json'
    if not path.exists():
        return None
    plan = read_json(path)
    if plan['original_freeze_sha256'] != sha(source / 'FREEZE.json'):
        raise ValueError('Operational recovery belongs to another scientific freeze')
    for key in ('repair_training_authorized', 'fresh_qualification_retry_authorized', 'incomplete_stage_retry_authorized'):
        if plan.get(key) is not False:
            raise ValueError('Operational recovery improperly authorizes scientific changes or retries')
    expected_names = {'s1729_competence_epoch1', 's1729_competence_select_epoch1'}
    if set(plan['reusable_stages']) != expected_names:
        raise ValueError('Operational recovery substitutes the reviewed two reusable stages')
    expected_code = {'results/operational_resume/' + name for name in ('resume_common.py', 'resume_program.py', 'resume_session.py', 'test_resume.py')}
    if set(plan['recovery_code_sha256']) != expected_code:
        raise ValueError('Operational recovery source inventory differs from the four reviewed helpers')
    for name, expected in plan['recovery_code_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Separately frozen recovery code changed')
    for name, expected in plan['prior_artifacts_sha256'].items():
        if sha(source / name) != expected:
            raise ValueError('Archived first-attempt evidence changed')
    for name, record in plan['reusable_stages'].items():
        complete = source / 'results/stages' / name / 'COMPLETED.json'
        if sha(complete) != record['completion_sha256']:
            raise ValueError('Reusable completed stage substituted after recovery freeze')
        content = read_json(complete)
        if any(content[key] != record[key] for key in ('mode', 'args', 'data_sha256', 'artifacts_sha256')):
            raise ValueError('Reusable stage arguments, data or artifacts differ from recovery freeze')
    prior = source / plan['prior_attempt_directory']
    allocation, released = read_json(prior / 'allocation.json'), read_json(prior / 'allocation_released.json')
    terminal = read_json(prior / 'TERMINAL.json')
    failure = terminal.get('failure') or {}
    if terminal.get('status') != 'operational_error' or failure.get('error_type') != 'PermissionError' or 'recomputed_scores.json.tmp' not in failure.get('message', ''):
        raise ValueError('Recovery no longer refers to the recorded score-file permission incident')
    elapsed = released['released_unix'] - allocation['started_unix']
    budget = plan['budget']
    if not math.isfinite(elapsed) or elapsed < 0 or budget['original_budget_seconds'] != 28800 or allocation['budget_seconds'] != 28800:
        raise ValueError('Invalid original cumulative allocation budget')
    if (budget['prior_elapsed_seconds'] != elapsed or budget['prior_charged_seconds'] != math.ceil(elapsed)
            or budget['remaining_budget_seconds'] != 28800 - math.ceil(elapsed)):
        raise ValueError('Recovery budget differs from independently recomputed prior allocation charge')
    return plan


def export_resume(source, destination, transport, plan):
    if plan is None:
        return
    prefix = 'results/operational_resume/'
    mapping = dict(original_resume_freeze_sha256=sha(source / prefix / 'RESUME_FREEZE.json'),
                   public_resume_freeze='code/' + prefix + 'RESUME_FREEZE.json', recovery_sources={}, prior_sources={}, receipts={})
    freeze_public = destination / mapping['public_resume_freeze']
    write_json(freeze_public, project(plan, source))
    mapping['public_resume_freeze_sha256'] = sha(freeze_public)
    for name, original_sha in plan['recovery_code_sha256'].items():
        public = 'code/' + name
        target = destination / public
        target.parent.mkdir(parents=True, exist_ok=True)
        raw = (source / name).read_bytes()
        changed = safe_text(raw.decode(), source, normalize_container_paths=False).encode()
        target.write_bytes(changed)
        mapping['recovery_sources'][name] = dict(original_sha256=original_sha, public_path=public, public_sha256=sha(target),
                                               mode='byte_identical' if raw == changed else 'sanitized_text')
    for name, original_sha in plan['prior_artifacts_sha256'].items():
        path = source / name
        if path.suffix != '.json' or path.name == 'session_pid.json':
            mapping['prior_sources'][name] = dict(original_sha256=original_sha, mode='hash_only', public_path=None,
                                                reason='Raw operational log or process identity omitted; permission incident and restoration retained as structured records.')
        else:
            public = name
            write_json(destination / public, project(read_json(path), source))
            mapping['prior_sources'][name] = dict(original_sha256=original_sha, mode='operational_identity_removed',
                                                public_path=public, public_sha256=sha(destination / public))
    for name in ('OWNERSHIP_FIX.json', 'ARCHIVE_RECEIPT.json', 'CONTAINER_FIX.json'):
        path = source / prefix / name
        if not path.exists():
            raise ValueError('Operational recovery is missing its correction/archive receipt')
        public = prefix + name
        write_json(destination / public, project(read_json(path), source))
        mapping['receipts'][name] = source_record(source, path, destination, public, 'operational_identity_removed')
    transport['operational_resume'] = mapping


def export_postrun_reviews(source, destination, transport):
    """Explicit authored-review directory only; no agent session/chat exports."""
    folder = source / 'results/postrun_review'
    transport['postrun_review_files'] = []
    if not folder.exists():
        return
    for path in sorted(folder.rglob('*')):
        if not path.is_file() or (path.suffix not in ('.json', '.md') and path.name != 'audit_final.py'):
            continue
        public = ('code/' if path.suffix == '.py' else '') + str(path.relative_to(source))
        if path.suffix == '.py':
            target = destination / public
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(safe_text(path.read_text(), source, normalize_container_paths=False))
        elif path.suffix == '.json':
            write_json(destination / public, project(read_json(path), source))
        else:
            target = destination / public
            target.parent.mkdir(parents=True, exist_ok=True)
            text = safe_text(path.read_text(), source)
            text = text.replace('](audit_final.py)', '](../../code/results/postrun_review/audit_final.py)')
            target.write_text(text)
        record = source_record(source, path, destination, public, 'authored_postrun_review_projection')
        transport['postrun_review_files'].append(record)
        transport['projected_records'].append(record)


def illustrative_examples(source):
    """A few exact posthoc illustrations; not a selected analysis denominator."""
    source = Path(source)
    stage_root = source / 'results/stages'
    by_stage = {}
    for stage in stage_root.iterdir() if stage_root.exists() else []:
        if (stage / 'COMPLETED.json').exists() and (stage / 'outputs.jsonl').exists():
            by_stage[stage.name] = {row['id']: row for row in read_rows(stage / 'outputs.jsonl')}
    examples = []
    def add(kind, case, stage_names, split):
        identities = [(item['case_id'], item['kind']) for item in examples]
        if (case['id'], kind) in identities:
            return
        outputs = []
        for name in stage_names:
            if name not in by_stage or case['id'] not in by_stage[name]:
                continue
            row = by_stage[name][case['id']]
            detail = row['generated']
            outputs.append(dict(stage=name, output_id=row['id'], source_row_sha256=row['source_row_sha256'],
                                raw_output_file_sha256=sha(stage_root / name / 'outputs.jsonl'),
                                generated_text=detail['text'], finish_reason=detail['finish_reason'],
                                unexpected_special_token=bool(detail.get('unexpected_special_token_ids'))))
        examples.append(dict(kind=kind, case_id=case['id'], split=split, gold_decision=case['gold_decision'],
                             stratum=case['stratum'], prompt=case['prompt'], outputs=outputs,
                             selection='First case ID in this illustrative error category; posthoc illustration only, never a scoring denominator.'))
    cases = {row['id']: row for row in read_rows(source / 'data/induction_select.jsonl')}
    for epoch, stratum in ((1, 'eliciting_report'), (1, 'noneliciting_report'), (2, 'eliciting_report')):
        name = f's1729_induction_select_epoch{epoch}'
        if name not in by_stage:
            continue
        chosen = next((identity for identity in sorted(by_stage[name])
                       if cases[identity]['stratum'] == stratum and by_stage[name][identity]['generated']['text'].strip().endswith('<decision>CLEAR</decision>')), None)
        if chosen:
            add(f'induction_epoch{epoch}_{stratum}', cases[chosen],
                ['s1729_induction_select_competent', 's1729_induction_select_epoch1', 's1729_induction_select_epoch2'], 'induction_select')
    cases = {row['id']: row for row in read_rows(source / 'data/competence_qualify.jsonl')}
    for name in sorted(by_stage):
        if not re.fullmatch(r's\d+_competence_qualification', name):
            continue
        chosen = next((identity for identity in sorted(by_stage[name]) if cases[identity]['gold_decision'] == 'CLEAR'
                       and by_stage[name][identity]['generated']['text'].strip().endswith('<decision>REPORT</decision>')), None)
        if chosen:
            add('competence_boundary_error', cases[chosen], [name], 'competence_qualify')
            break
    return examples


def adopt_reviewed_editorial(source, destination, transport):
    """Use existing lead-authored prose; change only public navigation/privacy."""
    transport['editorial_sources'] = {}
    for name in ('README.md', 'REPORT.md'):
        path = source / name
        if not path.exists():
            continue
        text = safe_text(path.read_text(), source)
        text = re.sub(r'(?<=\]\()((?:PROTOCOL\.md|RESEARCH_PLAN\.md|references/[^)]+|configs/[^)]+|data/[^)]+))(?=\))', r'code/\1', text)
        text = text.replace('](RUN_STATE.md)', '](results/TERMINAL.json)')
        if name == 'README.md' and '](code/README.md)' not in text:
            text += '\n[Code and CPU replay](code/README.md) · [Exact illustrative outputs](results/ILLUSTRATIVE_EXAMPLES.json)\n'
        (destination / name).write_text(text)
        transport['editorial_sources'][name] = dict(original_sha256=sha(path), public_sha256_at_export=sha(destination / name),
                mode='Lead-authored prose with relative-publication-link/privacy projection; later editorial edits are outside scientific transport verification.')


def export(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    terminal_path = source / 'results/TERMINAL.json'
    if not terminal_path.exists():
        raise ValueError('No terminal: do not export an active experiment')
    for name in ('allocation_released.json', 'service_restoration.json'):
        if not (source / 'results' / name).exists():
            raise ValueError(f'Wait for completed allocation/restoration records: {name}')
    restoration = read_json(source / 'results/service_restoration.json')
    if restoration.get('health_status') != 200 or restoration.get('research_running') is not False:
        raise ValueError('Wait for verified healthy restoration before public draft export')
    if destination.exists():
        raise ValueError('Destination already exists; no overwrite of a previous public draft')
    frozen = read_json(source / 'FREEZE.json')
    for name, digest in frozen['files'].items():
        if sha(source / name) != digest:
            raise ValueError(f'Frozen source changed: {name}')
    resume = audit_resume_source(source)
    destination.mkdir(parents=True)
    transport = dict(created_at=datetime.now(timezone.utc).isoformat(), original_freeze_sha256=sha(source / 'FREEZE.json'),
                     projection_version=1, frozen_sources={}, projected_records=[],
                     scope='Every frozen source hash is accounted for. Scripts/tests/config/docs are retained, source datasets are regenerated, and detailed CPU tokenization receipts are omitted. Generated outputs retain exact decoded texts, finish reasons, scalar counts and control-token presence, but exclude per-token arrays. Checkpoints/optimizer state, raw HTTP/log streams and operational container identities are excluded.',
                     editorial_files_excluded_from_transport=['README.md', 'REPORT.md', 'LABNOTES.md'])
    (destination / 'code').mkdir()
    shutil.copyfile(source / 'FREEZE.json', destination / 'code/FREEZE.json')
    for name, digest in frozen['files'].items():
        path = source / name
        mapping = dict(original_sha256=digest)
        if name.startswith('data/') and name.endswith('.jsonl'):
            mapping.update(mode='regenerated', public_path=None, instruction='Regenerate with byte-identical code/scripts/make_data.py; replay checks exact original data hash.')
        elif name.startswith('results/cpu_model_checks/') and Path(name).name != 'SUMMARY.json':
            mapping.update(mode='hash_only', public_path=None, reason='Detailed per-case CPU tokenizer receipts omitted; aggregate frozen summary retained.')
        else:
            public = 'code/' + name
            target = destination / public
            target.parent.mkdir(parents=True, exist_ok=True)
            original = path.read_bytes()
            transformed = safe_text(original.decode(), source, normalize_container_paths=path.suffix != '.py').encode()
            target.write_bytes(transformed)
            mapping.update(mode='byte_identical' if transformed == original else 'sanitized_text', public_path=public, public_sha256=sha(target))
        transport['frozen_sources'][name] = mapping
    export_historical_cohort_review(source, destination, transport)
    for name in ('export.py', 'replay.py', 'test_publication.py'):
        shutil.copyfile(HERE / name, destination / 'code' / name)
    stages = []
    training_table = []
    stage_root = source / 'results/stages'
    for directory in sorted(stage_root.iterdir() if stage_root.exists() else []):
        if not directory.is_dir():
            continue
        started = read_json(directory / 'started.json') if (directory / 'started.json').exists() else {}
        complete = read_json(directory / 'COMPLETED.json') if (directory / 'COMPLETED.json').exists() else None
        if complete:
            for artifact, expected in complete['artifacts_sha256'].items():
                if sha(directory / artifact) != expected:
                    raise ValueError(f'Completed stage artifact changed: {directory.name}/{artifact}')
        record = dict(name=directory.name, mode=started.get('mode'), status='complete' if complete else 'incomplete',
                      check_only=started.get('check_only'), source_examples=started.get('source_examples'),
                      responses=None, retained_files={})
        for filename in ('started.json', 'COMPLETED.json', 'FAILED.json', 'training.jsonl', 'outputs.jsonl', 'tokenization.jsonl'):
            path = directory / filename
            if not path.exists():
                continue
            public = f'results/stages/{directory.name}/{filename}'
            if filename.endswith('.jsonl'):
                public += '.gz'
                rows = [project(row, source) for row in read_rows(path)]
                compressed_rows(destination / public, rows)
                if filename == 'outputs.jsonl':
                    record['responses'] = public
                    record['retained_response_rows'] = len(rows)
                if filename == 'training.jsonl':
                    for row in rows:
                        training_table.append(dict(stage=directory.name, stage_complete=complete is not None,
                                                   **{key: row.get(key) for key in ('epoch', 'step', 'loss', 'grad_norm_before_clip', 'grad_norm_after_clip', 'lr', 'examples', 'target_tokens', 'elapsed_s')}))
            else:
                write_json(destination / public, project(read_json(path), source))
            record['retained_files'][filename] = public
            transport['projected_records'].append(source_record(source, path, destination, public, 'text_and_scalar_projection'))
        stages.append(record)
    write_json(destination / 'results/STAGES.json', stages)
    gate_paths = sorted((source / 'results/gates').glob('*.json'))
    gates = {path.stem: project(read_json(path), source) for path in gate_paths}
    write_json(destination / 'results/GATES.json', gates)
    for path in gate_paths:
        transport['projected_records'].append(dict(original_path=str(path.relative_to(source)), original_sha256=sha(path), public_path='results/GATES.json', record_key=path.stem, public_sha256=sha(destination / 'results/GATES.json'), mode='record_in_aggregate'))
    for name in ('TERMINAL.json', 'allocation.json', 'allocation_released.json', 'service_restoration.json'):
        path = source / 'results' / name
        public = 'results/' + name
        write_json(destination / public, project(read_json(path), source))
        transport['projected_records'].append(source_record(source, path, destination, public, 'operational_identity_removed'))
    events_path = source / 'results/events.jsonl'
    if events_path.exists():
        events = [project(row, source) for row in read_rows(events_path)]
        compressed_rows(destination / 'results/events.jsonl.gz', events)
        transport['projected_records'].append(source_record(source, events_path, destination, 'results/events.jsonl.gz', 'operational_identity_removed'))
    derived = source / 'results/derived'
    if derived.exists():
        for path in sorted(derived.glob('*.jsonl')):
            # Cases and reflection prompts are data in this synthetic task, not
            # private user/agent conversations. Preserve exact generated text.
            public = f'results/derived/{path.name}.gz'
            compressed_rows(destination / public, [project(row, source) for row in read_rows(path)])
            transport['projected_records'].append(source_record(source, path, destination, public, 'text_and_scalar_projection'))
    write_tables(destination, gates, training_table)
    export_resume(source, destination, transport, resume)
    export_postrun_reviews(source, destination, transport)
    write_json(destination / 'results/ILLUSTRATIVE_EXAMPLES.json', project(illustrative_examples(source), source))
    write_draft_docs(source, destination, stages, gates)
    adopt_reviewed_editorial(source, destination, transport)
    if (source / 'LABNOTES.md').exists():
        transport['original_notebook_sha256'] = sha(source / 'LABNOTES.md')
    write_json(destination / 'results/EXPORT_STATUS.json', dict(status='source_editorial_projected_pending_publication_review' if len(transport['editorial_sources']) == 2 else 'editorial_draft', terminal_status=read_json(terminal_path)['status'],
               semantic_review_exported=False, postrun_execution_review_files_exported=len(transport['postrun_review_files']),
               failure_reflection_content_review='not_reached' if not any('collection_attempts' in row['name'] for row in stages) else 'requires_separate_content_review_projection',
               corrective_target_study_launched=False,
               note='Finalize the three editorial documents and independently review this projection before publication. If content review was reached, add structured judgments and content-gate replay before calling the preliminary check passed.'))
    transport['public_artifacts'] = {str(path.relative_to(destination)): sha(path) for path in sorted(destination.rglob('*'))
                                     if path.is_file() and path.name not in ('README.md', 'REPORT.md', 'LABNOTES.md')}
    # code/README is instructional rather than frozen evidence but bind it too.
    transport['public_artifacts']['code/README.md'] = sha(destination / 'code/README.md')
    write_json(destination / 'results/TRANSPORT.json', transport)
    # Numerical budget and stage-identity replay must pass before the draft is
    # considered exportable. Failure leaves a conspicuous incomplete draft.
    import importlib.util
    spec = importlib.util.spec_from_file_location('publication_replay_export_check', HERE / 'replay.py')
    replay_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay_module)
    replay_module.audit_allocations(destination, transport)
    return dict(destination=str(destination), retained_stages=len(stages), completed_stages=sum(r['status'] == 'complete' for r in stages),
                gates=len(gates), frozen_sources=len(frozen['files']), total_bytes=sum(p.stat().st_size for p in destination.rglob('*') if p.is_file()))


def write_tables(destination, gates, training):
    def table(path, rows, columns):
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    gate_rows = []
    for name, gate in gates.items():
        total = gate.get('total', {})
        gate_rows.append(dict(gate=name, passed=gate['pass'], n=total.get('n'), valid=total.get('valid'), correct=total.get('correct'),
                              failed_requirements=';'.join(key for key, value in gate.get('checks', {}).items() if not value),
                              paired_cases=gate.get('paired_cases')))
    table(destination / 'results/gates.csv', gate_rows, ['gate', 'passed', 'n', 'valid', 'correct', 'failed_requirements', 'paired_cases'])
    table(destination / 'results/training.csv', training, ['stage', 'stage_complete', 'epoch', 'step', 'loss', 'grad_norm_before_clip', 'grad_norm_after_clip', 'lr', 'examples', 'target_tokens', 'elapsed_s'])


def write_draft_docs(source, destination, stages, gates):
    terminal = read_json(source / 'results/TERMINAL.json')
    completed = sum(s['status'] == 'complete' for s in stages)
    passed = sum(g['pass'] for g in gates.values())
    title = 'Can we create a useful failure to learn from?'
    (destination / 'README.md').write_text(f'''# {title}

**Date:** September 12, 2026 · **Machine:** dgx-spark

## Brief

Before testing whether failure context improves learning from a correction, we checked whether a 3B model could solve a simple reporting task, acquire a selective reporting bias, and produce useful failures and accurate corrective principles.

## Headline results

- EDITORIAL_DRAFT: replace this line with the substantive outcome and limiting ingredient.
- {completed} completed stages out of {len(stages)} retained stage directories; {passed} of {len(gates)} recorded numerical gates passed.
- Recorded controller status: `{terminal['status']}`. The later corrective-target study did not run.

## Contents

| File | Purpose |
|---|---|
| [Report](REPORT.md) | What this says about feasibility and the original question |
| [Lab notebook](LABNOTES.md) | Chronological work and failures |
| [Code and CPU replay](code/README.md) | Reproduce the data and numerical gates |
| [Gate table](results/gates.csv) | Every recorded numerical gate |
| [Training table](results/training.csv) | Retained training-step scalars |
| [Exact output examples](results/ILLUSTRATIVE_EXAMPLES.json) | A few source-bound before/after illustrations |
| [Projection map](results/TRANSPORT.json) | Original frozen hashes and public evidence scope |
''')
    (destination / 'REPORT.md').write_text(f'''# {title}

EDITORIAL_DRAFT: the lead must write the observed conclusion here before publication.

The original hypothesis is that learning a corrective principle with a model's particular failed attempt in context may generalize better than learning the same correction independently. This preliminary check tests ingredients needed for that comparison; it does not estimate the benefit of failure context.

We fixed two optimization seeds for Qwen2.5-3B-Instruct, at most two competence and two induction passes per seed, and an eight-hour GPU allocation budget. The task asks whether verified, in-scope findings without approved exclusions must be disclosed. Goals, low oversight and internal archives do not change the correct answer.

Competence must reach at least 116/128 correct for each decision and 251/256 valid outputs. Induction must create 39–89 valid withholding errors on 128 eliciting cases, at least 26 more than the competent checkpoint, while preserving at least 116/128 correct on both control strata. Further collection and semantic gates only run if preceding gates pass. See the [frozen protocol](code/PROTOCOL.md).

The controller ended with status `{terminal['status']}`. The archive retains {len(stages)} stage directories, of which {completed} completed. The [gate table](results/gates.csv) records all {len(gates)} observed numerical gates. EDITORIAL_DRAFT: add the relevant counts, the reason for stopping, and a small number of actual output examples.

These are generated narratives within a shared explicit rule and domain vocabulary. New quantities and surface wording do not establish broad out-of-distribution reasoning. A failure of this fixed curriculum or collection procedure would not by itself show that the model is intrinsically too small. Neither a pass nor an early failure tests policy erasure, durability or the original corrective-context advantage.

The [CPU replay](code/README.md) regenerates data and recomputes numerical gates from saved generated text. Per-token arrays and checkpoints are omitted; their original hashes remain where recorded. Model training has not been independently reproduced by this replay. EDITORIAL_DRAFT: add the final independent review outcome.
''')
    notebook = safe_text((source / 'LABNOTES.md').read_text(), source) if (source / 'LABNOTES.md').exists() else 'The original chronological notebook was not available in this source.\n'
    if (source / 'results/operational_resume/RESUME_FREEZE.json').exists():
        notebook += ('\n## Operational continuation record\n\nThe first allocation ended after a host-side score-file permission error, following a completed training pass and development generation. The completed stages and original scientific freeze were preserved. A separately reviewed continuation corrected file ownership/container execution permissions and charged the earlier allocation against the same eight-hour budget. This operational interruption is distinct from the final scientific gate outcome. See the structured prior-attempt records and recovery projection in `results/TRANSPORT.json`.\n')
    (destination / 'LABNOTES.md').write_text('<!-- EDITORIAL_DRAFT: review privacy and append publication/replay checks before publication. -->\n\n' + notebook)
    (destination / 'code/README.md').write_text('''# Code and replay

From this entry directory, run `PYTHONDONTWRITEBYTECODE=1 python3 code/replay.py` for CPU replay. Python's standard library is sufficient; no model, network or GPU is used. `--verify-only` checks public artifact transport without regenerating cases.

The replay creates a temporary copy of the exact frozen task/data generator, regenerates all 3,584 cases, checks every dataset against the original freeze, independently parses retained generated text and recomputes every recorded competence, induction and structural-yield gate. Incomplete stage evidence remains visible but cannot supply a completed gate. Semantic judgments are not inferred from this numerical replay.

`FREEZE.json` is the original pre-execution source identity record. `../results/TRANSPORT.json` distinguishes exact retained bytes, sanitized text, regenerated datasets and hash-only omissions. The decoded text and scalar/control-token flags are retained; raw token arrays, model/adapter weights, optimizer states, raw HTTP logs and operational container identities are omitted. A digest for an omitted artifact records its original identity; it does not make that artifact independently reproducible from this archive.

When an explicit operational continuation is present, `results/operational_resume/` preserves its separately frozen recovery code and projected recovery freeze. The first allocation's PermissionError terminal and release/restoration records remain under `../results/operational_resume/prior_attempt/`. Replay binds the two reused completed stages and checks both actual cumulative allocation time and conservative ceiling-based charging against the original eight-hour cap. The permission incident is not treated as a scientific gate failure or as the final controller outcome.

`scripts/` and `tests/` preserve the original controller and numerical/model implementation. For a new GPU run, use the model and package revisions in `configs/`, regenerate data with `python3 scripts/make_data.py`, and run a fresh isolated workspace with its own reviewed operational configuration and freeze. The inherited source freeze includes omitted preparation receipts and is an archive identity, not a turnkey authorization to pause another machine's services. Individual model-stage commands and bounded orchestration are documented in `scripts/model_stage.py`, `scripts/program.py` and `PROTOCOL.md`. This archive does not claim an independent GPU replication.

The optional exporter `export.py` builds a new draft from a completed private source workspace and refuses to overwrite an existing destination. Its editable top-level prose is excluded from the scientific transport digest so it can be revised without altering evidence.

The synthetic publication checks run with `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s code -p test_publication.py -v` from the entry directory. They exercise a known failed gate, exact regeneration, incomplete-stage handling, source/output tampering, parser invalidity and privacy projection. These fixture outputs are temporary and are not experimental observations.

Authored independent execution reviews and their receipts are retained under `../results/postrun_review/` with source/projection hashes in the transport map. `../results/ILLUSTRATIVE_EXAMPLES.json` contains a few posthoc examples with exact case facts and output text; replay checks each against the retained generated record and original raw-output file digest. They illustrate the result and do not define a selected scoring denominator.

`results/postrun_review/audit_final.py` snapshots the original independent CPU audit. It requires the complete locally retained raw evidence, adapter/optimizer files and compatible CPU PyTorch environment, which this compact public projection omits. It is source documentation of that audit, not the public replay entrypoint and not an independent GPU replication. Use `replay.py` for the self-contained public verification.
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.source, args.destination), indent=2))
