"""Deterministic, factual-grounded donor pairing and globally balanced repairs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

try:
    from .task import canonical_json, donor_key, gold, parse_output, policy_family
    from .serialization import answer_text, render_case, serialized_case, ORDERS
except ImportError:
    from task import canonical_json, donor_key, gold, parse_output, policy_family
    from serialization import answer_text, render_case, serialized_case, ORDERS

class CohortGateFailure(ValueError):
    """A scientifically insufficient actual-failure cohort, not invalid provenance."""


ARMS = ('matched_failure', 'donor_failure', 'context_only', 'correct_trace', 'direct')
SYSTEM = 'You audit small ledgers against their stated policy. Return exactly the requested JSON object. Follow the policy and the current case facts.'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def construct(pool_path, outputs_path, preservation_path, destination, seed, output_order="decision_first"):
    if output_order not in ORDERS:
        raise ValueError("Unknown output serialization")
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Cannot overwrite a cohort: {destination}')
    pool = {row['id']: row for row in read(pool_path)}
    outputs = read(outputs_path)
    if len(outputs) != len(pool) or {o['id'] for o in outputs} != set(pool):
        raise ValueError('Failure collection is not complete for the fixed pool.')
    eligible = []
    reasons = Counter()
    for output in outputs:
        row = pool[output['id']]
        source_hash = hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                              ensure_ascii=False).encode()).hexdigest()
        if output.get('source_row_sha256') != source_hash:
            raise ValueError('Collected failure is not bound to this exact source row.')
        rendered_hash = hashlib.sha256(json.dumps(serialized_case(row, output_order), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()
        if output.get('rendered_row_sha256') != rendered_hash:
            raise ValueError('Collected failure is not bound to the planned serialized prompt and target.')
        raw = output['generated']['text']
        parsed = parse_output(raw, row)
        if parsed != output['parsed']:
            raise ValueError('Saved parser result changed under current oracle.')
        if output['generated']['finish_reason'] != 'eos':
            reasons['unfinished_generation'] += 1
            continue
        if not parsed['eligible_failure']:
            reasons['not_verified_supported_failure'] += 1
            continue
        eligible.append(dict(id=row['id'], case=row, trace=raw,
                             canonical_trace=canonical_json(parsed['parsed']),
                             operator=parsed['operator']))
    groups = defaultdict(list)
    for item in eligible:
        groups[donor_key(item['case'], item['operator'])].append(item)
    rng = random.Random(seed)
    units = []
    for key in sorted(groups):
        by_text = defaultdict(list)
        for item in groups[key]:
            by_text[item['canonical_trace']].append(item)
        for values in by_text.values():
            rng.shuffle(values)
        # Pair the two largest remaining distinct trace groups. This maximizes
        # the number of disjoint nonidentical pairs within each exact donor cell.
        while True:
            options = sorted(((-len(v), k) for k, v in by_text.items() if v))
            if len(options) < 2:
                break
            a, b = (by_text[options[i][1]].pop() for i in range(2))
            units.append((a, b))
    rng.shuffle(units)
    chosen = []
    used = set()
    family_counts = Counter()
    operator_counts = Counter()

    def take(index):
        used.add(index)
        chosen.append(units[index])
        family_counts[policy_family(units[index][0]['case'])] += 2
        operator_counts[units[index][0]['operator']] += 2

    families = sorted({policy_family(row) for row in pool.values()})
    if len(families) != 8:
        raise ValueError('Pool does not contain all eight policy families.')
    for family in sorted(families, key=lambda f: sum(policy_family(u[0]['case']) == f for u in units)):
        candidates = [i for i, pair in enumerate(units)
                      if i not in used and policy_family(pair[0]['case']) == family]
        if not candidates:
            raise CohortGateFailure(f'No nonidentical actual donor pair for family {family}')
        take(candidates[0])
    for operator in ('recent_only', 'requested_waivers', 'collapse_issues'):
        for index, pair in enumerate(units):
            if operator_counts[operator] >= 32:
                break
            if index not in used and pair[0]['operator'] == operator:
                take(index)
        if operator_counts[operator] < 32:
            raise CohortGateFailure(f'Fewer than 32 matchable failures for operator {operator}')
    for index in range(len(units)):
        if len(chosen) >= 96:
            break
        if index not in used:
            take(index)
    if len(chosen) != 96:
        raise CohortGateFailure(f'Only {2 * len(chosen)} failures can form the required closed cohort of 192.')
    paired = []
    for a, b in chosen:
        if a['canonical_trace'] == b['canonical_trace'] or a['id'] == b['id']:
            raise ValueError('Invalid donor pair')
        for current, donor in ((a, b), (b, a)):
            paired.append(dict(current=current, donor=donor))
    preservation = read(preservation_path)
    preserve = []
    for family in families:
        for decision, count in (('REPORT', 8), ('CLEAR', 32)):
            candidates = [r for r in preservation
                          if policy_family(r) == family and gold(r)['decision'] == decision]
            if len(candidates) < count:
                raise ValueError(f'Insufficient preservation cells: {family}/{decision}')
            # Frozen source order, independent of any model output.
            preserve.extend(candidates[:count])
    if len(preserve) != 320:
        raise ValueError('Preservation must contain 64 REPORT and 256 CLEAR examples.')
    destination.mkdir(parents=True)
    mapping = []
    for pair in paired:
        a, b = pair['current'], pair['donor']
        mapping.append(dict(id=a['id'], donor_id=b['id'], operator=a['operator'],
                            donor_cell=list(donor_key(a['case'], a['operator'])),
                            trace=a['trace'], donor_trace=b['trace'], target=answer_text(a['case'], output_order, target=gold(a['case'])),
                            current_case=a['case'], donor_case=b['case']))
    write(destination / 'matched_cases.jsonl', mapping)
    for arm in ARMS:
        training = []
        for item in mapping:
            row = item['current_case']
            current = 'CURRENT CASE ' + row['id'] + '\n' + render_case(row, output_order)
            current += '\nAudit this current case under its own stated policy. An archived attempt, if provided, may be wrong.'
            msgs = [{'role': 'system', 'content': SYSTEM}]
            if arm != 'direct':
                archive = item['donor_case'] if arm == 'donor_failure' else row
                attempt = (item['donor_trace'] if arm == 'donor_failure' else item['trace']
                           if arm == 'matched_failure' else answer_text(row, output_order, target=gold(row))
                           if arm == 'correct_trace' else 'No attempted answer was recorded for this archived case.')
                msgs.extend([{'role': 'user', 'content': 'ARCHIVED CASE ' + archive['id'] + '\n' + render_case(archive, output_order)},
                             {'role': 'assistant', 'content': attempt}])
            msgs.append({'role': 'user', 'content': current})
            training.append(dict(id=row['id'], kind='failure_correction', messages=msgs,
                                 target=item['target'], source_id=row['id'], output_order=output_order))
        for row in preserve:
            training.append(dict(id=row['id'], kind='preservation', source_id=row['id'],
                messages=[{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': render_case(row, output_order)}],
                target=answer_text(row, output_order, target=gold(row)), output_order=output_order))
        if len(training) != 512 or len({r['id'] for r in training}) != 512:
            raise ValueError('Training examples are not unique and complete.')
        labels = Counter(json.loads(r['target'])['decision'] for r in training)
        if labels != Counter(REPORT=256, CLEAR=256):
            raise ValueError('Global target balance failed.')
        write(destination / f'{arm}.jsonl', training)
    reference = read(destination / 'matched_failure.jsonl')
    for arm in ARMS:
        actual = read(destination / f'{arm}.jsonl')
        if [(r['id'], r['target']) for r in actual] != [(r['id'], r['target']) for r in reference]:
            raise ValueError('Targets or case order differ between arms.')
        if actual[192:] != reference[192:]:
            raise ValueError('Preservation examples differ between arms.')
    summary = dict(status='complete', output_order=output_order, pool_examples=len(pool), collected_outputs=len(outputs),
        verified_supported_failures=len(eligible), exclusions=dict(reasons),
        available_nonidentical_disjoint_pairs=len(units), selected_failures=192,
        operator_counts=dict(operator_counts), policy_family_counts=dict(family_counts),
        preservation_examples=320, total_training_examples=512,
        target_decisions={'REPORT': 256, 'CLEAR': 256}, canonical_identical_donor_pairs=0,
        seed=seed, pool_sha256=digest(pool_path), outputs_sha256=digest(outputs_path),
        preservation_sha256=digest(preservation_path), script_sha256=digest(__file__),
        artifacts_sha256={p.name: digest(p) for p in sorted(destination.glob('*.jsonl'))})
    (destination / 'COMPLETED.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary
