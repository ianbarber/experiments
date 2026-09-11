"""CPU tokenizer and synthetic-cohort checks; no model measurements."""
import hashlib
import json
from pathlib import Path
import tempfile

from common import ROOT, SYSTEM, CONFIG, tokenizer, encode_example
from cohort import ARMS, construct
from serialization import ORDERS, serialized_case, transform_target
from task import bad_options, parse_output


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def row_sha(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def main():
    tok = tokenizer()
    suites, arms = {}, {}
    for order in ORDERS:
        suites[order] = {}
        for name in ('competence', 'calibration', 'validation', 'induction', 'pool', 'preservation', 'evaluation'):
            rows = [serialized_case(row, order) for row in read(ROOT / 'data' / f'{name}.jsonl')]
            encoded = [encode_example(tok, [{'role': 'system', 'content': SYSTEM},
                                           {'role': 'user', 'content': row['prompt']}], row['target']) for row in rows]
            suites[order][name] = dict(cases=len(rows), max_sequence_tokens=max(len(e['input_ids']) for e in encoded),
                                      min_target_tokens=min(e['target_tokens'] for e in encoded),
                                      max_target_tokens=max(e['target_tokens'] for e in encoded),
                                      total_target_tokens=sum(e['target_tokens'] for e in encoded))
            if suites[order][name]['max_target_tokens'] > CONFIG['generation_max_new_tokens']:
                raise ValueError('A supervised response cannot fit the generation cap.')
        pool = read(ROOT / 'data/pool.jsonl')
        simulated = []
        for row in pool:
            text = transform_target(bad_options(row)[row['designated_operator']], order)
            simulated.append(dict(id=row['id'], source_row_sha256=row_sha(row),
                                  rendered_row_sha256=row_sha(serialized_case(row, order)),
                                  generated={'text': text, 'finish_reason': 'eos'}, parsed=parse_output(text, row)))
        arms[order] = {}
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / 'synthetic_outputs.jsonl'
            output.write_text(''.join(json.dumps(row) + '\n' for row in simulated))
            fixture = construct(ROOT / 'data/pool.jsonl', output, ROOT / 'data/preservation.jsonl',
                                directory / 'cohort', 1729, output_order=order)
            reference_targets = None
            for arm in ARMS:
                rows = read(directory / 'cohort' / f'{arm}.jsonl')
                if any(row['output_order'] != order for row in rows):
                    raise ValueError('Cohort lost serialization identity.')
                encoded = [encode_example(tok, row['messages'], row['target']) for row in rows]
                targets = [[label for label in e['labels'] if label != -100] for e in encoded]
                if reference_targets is None:
                    reference_targets = targets
                if targets != reference_targets:
                    raise ValueError('Supervised token targets differ across repair arms.')
                for row, target in zip(rows, targets):
                    if target != tok.encode(row['target'], add_special_tokens=False) + [tok.eos_token_id]:
                        raise ValueError('Target-only label alignment failed.')
                arms[order][arm] = dict(cases=len(rows), target_tokens=sum(e['target_tokens'] for e in encoded),
                                       prefix_tokens=sum(e['prefix_tokens'] for e in encoded),
                                       max_sequence_tokens=max(len(e['input_ids']) for e in encoded))
            if fixture['selected_failures'] != 192:
                raise ValueError('Synthetic pairing fixture incomplete.')
            # The same source facts must not accept outputs from another order.
            wrong_order = ORDERS[1 - ORDERS.index(order)]
            try:
                construct(ROOT / 'data/pool.jsonl', output, ROOT / 'data/preservation.jsonl',
                          directory / 'wrong_order', 1729, output_order=wrong_order)
            except ValueError as exc:
                if 'serialized' not in str(exc):
                    raise
            else:
                raise AssertionError('Mismatched rendered output identity was accepted.')
    record = dict(status='passed', scope='CPU tokenizer and explicitly synthetic ideal-output cohort fixtures only; no model weights loaded or generated outputs.',
                  datasets=suites, synthetic_repair_arms=arms, identical_encoded_targets_within_order=True,
                  incorrect_serialization_provenance_rejected=True,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((ROOT / 'scripts').glob('*.py'))})
    with (ROOT / 'results/encoding_preflight.json').open('x') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
