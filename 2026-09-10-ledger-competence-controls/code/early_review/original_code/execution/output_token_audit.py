"""Decode saved IDs with the pinned tokenizer only; no model construction."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from transformers import AutoTokenizer
import torch

assert not torch.cuda.is_initialized()
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
review = json.loads((HERE / 'CONTRACT_AUDIT.json').read_text())
tok = AutoTokenizer.from_pretrained('/models/Qwen2.5-3B-Instruct', local_files_only=True)
result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'no_model_constructed': True, 'stages': []}
for stage in review['stages']:
    if stage['mode'] != 'generate':
        continue
    contract_path = ROOT / stage['path']
    assert hashlib.sha256(contract_path.read_bytes()).hexdigest() == stage['sha256']
    contract = json.loads(contract_path.read_text())
    outputs_path = contract_path.parent / 'outputs.jsonl'
    assert hashlib.sha256(outputs_path.read_bytes()).hexdigest() == contract['artifacts_sha256']['outputs.jsonl']
    rows = [json.loads(line) for line in outputs_path.read_text().splitlines()]
    for row in rows:
        generated = row['generated']
        assert tok.decode(generated['token_ids'], skip_special_tokens=True) == generated['text']
    result['stages'].append({'path': stage['path'], 'outputs': len(rows),
                            'minimum_tokens': min(r['generated']['generated_tokens'] for r in rows),
                            'maximum_tokens': max(r['generated']['generated_tokens'] for r in rows),
                            'total_tokens': sum(r['generated']['generated_tokens'] for r in rows)})
result['total_outputs'] = sum(s['outputs'] for s in result['stages'])
result['cuda_initialized'] = torch.cuda.is_initialized()
assert not result['cuda_initialized']
result['status'] = 'passed'
print(json.dumps(result, indent=2))
