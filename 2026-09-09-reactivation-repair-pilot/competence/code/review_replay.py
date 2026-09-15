"""Replay portable early-validity witnesses, never model outputs as code."""
from __future__ import annotations
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile

from export import encoded, read, require, sha_bytes
import replay

BASE = 'results/early_validity_review/'


def check_materialized(root, manifest, payloads, analyzer):
    packet = json.loads(payloads['__review__/REVIEW_PACKET.json'])
    require(packet['version'] == 1, 'Unknown early-review packet schema')
    require(packet['frozen_program_sha256'] == sha_bytes(payloads['results/FROZEN_PROGRAM.json']),
            'Review packet is for a different scientific freeze')
    for logical, record in packet['exact_artifacts'].items():
        require(record['operation'] == 'exact_original_review_artifact'
                and sha_bytes(payloads[logical]) == record['original_sha256'] == record['published_sha256'],
                'Exact historical review artifact differs: ' + logical)
    require(packet['portable_report_sha256'] == sha_bytes(payloads['__review__/REPORT.md']),
            'Authored portable review prose changed')
    require(packet['portable_amendment']['published_sha256'] == sha_bytes(payloads['__review__/SCOPE_AMENDMENT.md'])
            and packet['portable_amendment']['original_bytes_reproduced'] is False,
            'Edited portable amendment identity/scope differs')
    for logical, record in packet['omitted_original_prose'].items():
        require(record['original_bytes_reproduced'] is False, 'Original prose scope is misrepresented')
    root = Path(root)
    checker = replay.load_module('public_early_review_independent_checker', root / BASE / 'data/audit.py')
    counts = {'all_case_oracles': 0, 'authored_parser_witnesses': 0,
              'actual_text_witnesses': 0, 'authored_donor_sides': 0}
    projection_path = root / 'PUBLIC_PROJECTION.json'
    projections = read(projection_path)['files'] if projection_path.is_file() else {}
    actual_output_cache = {}
    with analyzer.frozen_modules(root / manifest['source_snapshot']) as modules:
        task = modules['task']
        by_id = {}
        for split in ('competence', 'calibration', 'validation', 'induction', 'pool', 'preservation', 'evaluation'):
            for case in [json.loads(line) for line in (root / f'data/{split}.jsonl').read_text().splitlines()]:
                require(checker.independent_execute(case) == task.gold(case), 'Independent oracle differs')
                by_id[case['id']] = case
                counts['all_case_oracles'] += 1
        for name, expected_field, counter in (
            ('parser_counterexamples.json', 'flags', 'authored_parser_witnesses'),
            ('actual_schema_invalid_credit_witnesses.json', 'independent_flags', 'actual_text_witnesses')):
            for item in json.loads(payloads[BASE + 'data/' + name])['cases']:
                case = item['case']
                require(by_id[case['id']] == case, 'Witness facts differ from regenerated data')
                flags = checker.independent_flags(item['raw_text'], case)
                require(flags == item[expected_field], 'Independent saved witness flags differ')
                parsed = task.parse_output(item['raw_text'], case)
                require(all(parsed[k] == v for k, v in flags.items()), 'Frozen parser witness differs')
                if counter == 'actual_text_witnesses':
                    logical = 'results/development/' + item['stage'] + '/outputs.jsonl'
                    if logical not in actual_output_cache:
                        data = (root / logical).read_bytes()
                        saved_rows = [json.loads(line) for line in data.splitlines() if line.strip()]
                        if logical in projections:
                            declaration = projections[logical]
                            require(sha_bytes(data) == declaration['projected_sha256'], 'Projected witness source bytes differ')
                            original_sha = declaration['original_sha256']
                            row_hashes = declaration['row_original_sha256']
                        else:
                            original_sha = sha_bytes(data)
                            row_hashes = [checker.row_sha(row) for row in saved_rows]
                        require(len(row_hashes) == len(saved_rows), 'Witness source row identity coverage differs')
                        actual_output_cache[logical] = (original_sha, {row['id']: (row, h) for row, h in zip(saved_rows, row_hashes)})
                    original_sha, saved_rows = actual_output_cache[logical]
                    original_row, recorded_row_sha = saved_rows[item['id']]
                    require(original_sha == item['outputs_sha256'] and recorded_row_sha == item['original_row_sha256'],
                            'Historical actual witness source identity differs')
                    require(original_row['generated']['text'] == item['raw_text']
                            and original_row['generated']['finish_reason'] == item['finish_reason'],
                            'Historical actual witness text differs from saved measurement')
                counts[counter] += 1
        witness = json.loads(payloads[BASE + 'design/EVIDENCE.json'])['donor_witness']
        for label in ('current', 'donor'):
            item = witness[label]; case = item['case']
            require(by_id[case['id']] == case and task.gold(case) == item['gold_audit'], 'Donor witness facts differ')
            require(task.parse_output(task.canonical_json(item['wrong_audit']), case)['eligible_failure'],
                    'Authored donor program no longer satisfies the declared witness')
            counts['authored_donor_sides'] += 1
        require(witness['current']['case']['id'] != witness['donor']['case']['id'], 'Donor witness is a self-pair')
        require(witness['current']['wrong_audit'] != witness['donor']['wrong_audit'], 'Donor witness is identical')
        adjudication = json.loads(payloads[BASE + 'data/PRACTICAL_ADJUDICATION.json'])
        preservation = [by_id[i] for i in adjudication['preservation_evidence']['selected_ids']]
        require(len(preservation) == 320 and len({r['id'] for r in preservation}) == 320,
                'Prospective preservation witness coverage differs')
        report = sum(task.gold(r)['decision'] == 'REPORT' for r in preservation)
        clear = len(preservation) - report
        trigger_report = sum(task.is_trigger(r) and task.gold(r)['decision'] == 'REPORT' for r in preservation)
        trigger_clear = sum(task.is_trigger(r) and task.gold(r)['decision'] == 'CLEAR' for r in preservation)
        require((report, clear, trigger_report, trigger_clear) == (64, 256, 0, 64),
                'Proposed repair routing witness differs')
        require(adjudication['heuristic']['correct'] == 192 + clear == 448,
                'Authored decision-only routing witness changed')
        counts['prospective_preservation_cases'] = len(preservation)
        counts['authored_routing_decisions_correct_of_512'] = 192 + clear
    return {'status': 'passed', 'checks': counts,
            'original_review_artifact_bytes_verified': len(packet['exact_artifacts']),
            'edited_prose_is_original_bytes': False,
            'actual_witness_raw_output_hashes': 'Recorded original identities; omitted token IDs are not reconstructed.',
            'historical_gpu_tokenizer_and_weight_audits_rerun': False,
            'new_model_calls': 0, 'new_gpu_calls': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    sys.dont_write_bytecode = True
    manifest, payloads = replay.verify_bundle(args.entry)
    with tempfile.TemporaryDirectory(prefix='public-early-review-replay-') as temporary:
        root = Path(temporary)
        analyzer = replay.materialize(root, manifest, payloads)
        print(json.dumps(check_materialized(root, manifest, payloads, analyzer), indent=2))


if __name__ == '__main__':
    main()
