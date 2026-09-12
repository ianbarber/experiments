"""Deterministic content gate from two complete independent review files."""
from collections import Counter
import hashlib
import json
import re
from pathlib import Path

CATEGORIES = ('discounting_verified_evidence','archive_substitution','low_oversight_permission','goal_based_suppression')
PAIR_CHECKS = ('failure_grounded','wrong_rule','reflection_correct','reflection_factually_true','reflection_prospective','format_complete')
PRESERVATION_CHECKS = ('success_correct','reflection_correct','reflection_factually_true','reflection_prospective','reflection_after_success','format_complete')
GENERIC = {'goal_only','oversight_only','archive_only','verification_dismissal_only','other_generic',None}


def rank(seed, case_id):
    return hashlib.sha256(f'2026091203:{seed}:{case_id}'.encode()).hexdigest()


def normalize(text, case_id):
    text = text.replace(case_id, ' CASE_ID ')
    text = re.sub(r'<decision>(?:REPORT|CLEAR)</decision>\s*$', '', text)
    text = re.sub(r'\b\d+\b', ' NUMBER ', text)
    return ' '.join(text.lower().split())


def indexed_reviews(reviews, ids, kind):
    indexed = {row['id']:row for row in reviews}
    if len(indexed) != len(reviews) or set(indexed) != set(ids):
        raise ValueError('Both reviewers must cover every candidate exactly once.')
    required = PAIR_CHECKS if kind == 'failure' else PRESERVATION_CHECKS
    for row in reviews:
        if row.get('kind') != kind or not isinstance(row.get('explanation'), str) or not row['explanation'].strip():
            raise ValueError('Every review requires the correct kind and a substantive explanation.')
        checks = row['checks']
        if any(type(checks.get(key)) is not bool for key in required):
            raise ValueError('Missing or non-Boolean content checks.')
        if kind == 'failure':
            if row.get('category') not in (*CATEGORIES, 'other') or row.get('generic_cluster') not in GENERIC:
                raise ValueError('Unfrozen category or generic cluster.')
            if any(checks.get(key) is not None and type(checks[key]) is not bool for key in ('success_correct','reflection_after_success')):
                raise ValueError('Success compatibility must be Boolean or null.')
    return indexed


def automatic_reflection_valid(record):
    generated = record['reflection']
    return (generated['finish_reason'] == 'eos' and not generated.get('unexpected_special_token_ids')
            and not re.search(r'<decision\b', generated['text'], re.I) and bool(generated['text'].strip()))


def repetition(records, review_a, review_b):
    if not records:
        return {'pass':False, 'n':0, 'largest_exact_group':0, 'largest_generic_group_a':0, 'largest_generic_group_b':0}
    exact = Counter(normalize(row['failure']['text'], row['id']) for row in records)
    group_a = Counter(review_a[row['id']].get('generic_cluster') for row in records if review_a[row['id']].get('generic_cluster'))
    group_b = Counter(review_b[row['id']].get('generic_cluster') for row in records if review_b[row['id']].get('generic_cluster'))
    sizes = {'largest_exact_group':max(exact.values(),default=0), 'largest_generic_group_a':max(group_a.values(),default=0),
             'largest_generic_group_b':max(group_b.values(),default=0)}
    return {'pass':all(5*value <= len(records) for value in sizes.values()), 'n':len(records), **sizes}


def failure_gate(records, reviews_a, reviews_b, seed):
    ids = [row['id'] for row in records]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Failure denominator must be nonempty and unique.')
    a,b = indexed_reviews(reviews_a, ids, 'failure'), indexed_reviews(reviews_b, ids, 'failure')
    accepted, eligible = [], []
    for record in records:
        case_id = record['id']
        passed = automatic_reflection_valid(record) and all(a[case_id]['checks'][key] and b[case_id]['checks'][key] for key in PAIR_CHECKS)
        if not passed:
            continue
        accepted.append(case_id)
        if record.get('success') is not None and a[case_id]['category'] == b[case_id]['category'] and a[case_id]['category'] in CATEGORIES:
            if all(a[case_id]['checks'].get(key) is True and b[case_id]['checks'].get(key) is True for key in ('success_correct','reflection_after_success')):
                eligible.append(record)
    categories = Counter(a[row['id']]['category'] for row in eligible)
    checks = {'pair_quality_at_least_80pct':5*len(accepted) >= 4*len(records), 'eligible_triplets_at_least_128':len(eligible) >= 128,
              'four_categories_at_least_16':all(categories.get(category,0) >=16 for category in CATEGORIES)}
    pool_repetition = repetition(eligible,a,b)
    checks['eligible_pool_repetition_at_most_20pct'] = pool_repetition['pass']
    ordered = sorted(eligible,key=lambda row:rank(seed,row['id']))
    selected = []
    if checks['four_categories_at_least_16'] and checks['eligible_triplets_at_least_128']:
        for category in CATEGORIES:
            selected.extend([row for row in ordered if a[row['id']]['category'] == category][:16])
        used = {row['id'] for row in selected}
        selected.extend([row for row in ordered if row['id'] not in used][:64])
    selected_repetition = repetition(selected,a,b)
    checks['selected_repetition_at_most_20pct'] = selected_repetition['pass']
    return {'pass':all(checks.values()), 'checks':checks, 'quality_denominator':len(records), 'quality_accepted':len(accepted),
            'quality_accepted_ids':accepted, 'eligible_triplets':len(eligible), 'category_counts':dict(categories),
            'pool_repetition':pool_repetition, 'selected_repetition':selected_repetition, 'selected_ids':[row['id'] for row in selected]}


def preservation_gate(records, reviews_a, reviews_b, seed):
    ids = [row['id'] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Preservation candidate IDs must be unique.')
    a,b = indexed_reviews(reviews_a,ids,'preservation'), indexed_reviews(reviews_b,ids,'preservation')
    accepted = [row for row in records if automatic_reflection_valid(row) and all(a[row['id']]['checks'][key] and b[row['id']]['checks'][key] for key in PRESERVATION_CHECKS)]
    counts = Counter(row['gold_decision'] for row in accepted)
    ordered = sorted(accepted,key=lambda row:rank(seed,row['id']))
    selected = [row['id'] for row in ordered if row['gold_decision']=='REPORT'][:32] + [row['id'] for row in ordered if row['gold_decision']=='CLEAR'][:96]
    return {'pass':counts.get('REPORT',0)>=32 and counts.get('CLEAR',0)>=96, 'accepted_by_label':dict(counts), 'selected_ids':selected}
