"""CPU-only exact tokenizer audit; no language-model construction or inference."""
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'scripts'))
from serialization import serialized_case
from transformers import AutoTokenizer
import torch

assert not torch.cuda.is_initialized()
tok = AutoTokenizer.from_pretrained('/models/Qwen2.5-3B-Instruct', local_files_only=True)
source = ast.parse((ROOT / 'scripts/common.py').read_text())
system = next(ast.literal_eval(n.value) for n in source.body if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'SYSTEM' for t in n.targets))
namespace = {'CONFIG': {'max_sequence_length': 2048}}
functions = [n for n in source.body if isinstance(n, ast.FunctionDef)
             and n.name in ('prefix_ids', 'encode_example')]
exec(compile(ast.Module(body=functions, type_ignores=[]), 'frozen_common_functions', 'exec'), namespace)
report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'no_model_constructed': True,
          'splits': {}, 'competence_by_id': {}}
all_missing = Counter()
for split in ('competence', 'calibration', 'validation', 'induction', 'pool', 'preservation', 'evaluation'):
    rows = [json.loads(line) for line in (ROOT / f'data/{split}.jsonl').read_text().splitlines()]
    stats = {order: Counter() for order in ('decision_first', 'decision_last')}
    unequal_prefix_lengths = unequal_target_lengths = unequal_prefix_sets = 0
    difference_tokens = Counter()
    missing_gold_target_token_counts = Counter()
    for row in rows:
        variants = []
        for order in stats:
            rendered = serialized_case(row, order)
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': rendered['prompt']}]
            prefix = namespace['prefix_ids'](tok, messages)
            encoded = namespace['encode_example'](tok, messages, rendered['target'])
            target = tok.encode(rendered['target'], add_special_tokens=False) + [tok.eos_token_id]
            variants.append((prefix, target, encoded))
            s = stats[order]
            s.update(examples=1, prefix_tokens=len(prefix), target_tokens=len(target))
            s['max_sequence_tokens'] = max(s['max_sequence_tokens'], len(encoded['input_ids']))
            s[f'{json.loads(rendered["target"])["decision"]}_target_tokens'] += len(target)
            for token in set(target) - set(prefix):
                missing_gold_target_token_counts[(order, token)] += 1
            if split == 'competence' and order == 'decision_first':
                report['competence_by_id'][row['id']] = {
                    'target_tokens': len(target), 'prefix_tokens': len(prefix),
                    'target_decision': json.loads(rendered['target'])['decision']}
        first, last = variants
        unequal_prefix_lengths += len(first[0]) != len(last[0])
        unequal_target_lengths += len(first[1]) != len(last[1])
        unequal_prefix_sets += set(first[0]) != set(last[0])
        difference_tokens.update(set(first[0]) ^ set(last[0]))
    report['splits'][split] = {
        'orders': stats, 'unequal_prefix_lengths': unequal_prefix_lengths,
        'unequal_target_lengths': unequal_target_lengths, 'unequal_prefix_token_sets': unequal_prefix_sets,
        'prefix_set_difference_tokens': [dict(token_id=k, text=tok.decode([k]), cases=v)
                                         for k, v in sorted(difference_tokens.items())],
        'target_tokens_absent_from_prefix': [dict(order=o, token_id=k, text=tok.decode([k]), cases=v)
                                            for (o, k), v in sorted(missing_gold_target_token_counts.items())]}
report['cuda_initialized_at_end'] = torch.cuda.is_initialized()
assert not report['cuda_initialized_at_end']
report['tokenizer'] = {'eos_token_id': tok.eos_token_id, 'pad_token_id': tok.pad_token_id,
                       'bos_token_id': tok.bos_token_id}
print(json.dumps(report, indent=2))
