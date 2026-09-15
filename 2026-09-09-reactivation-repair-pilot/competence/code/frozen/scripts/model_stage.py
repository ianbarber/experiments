"""Recorded GPU stages for the ledger follow-up; immutable outputs, no service control."""
import argparse
import gc
import json
import math
from pathlib import Path
import random
import time

import torch
from common import (ROOT, CONFIG, SYSTEM, now, sha, read_jsonl, tokenizer,
                    load_model, prefix_ids, encode_example, collate, token_losses,
                    generate_batch, seed_all)
from task import parse_output
from serialization import serialized_case
from objective import class_weight, effective_denominator, weighted_microbatch_loss


def messages(row):
    return row.get('messages') or [
        {'role': 'system', 'content': SYSTEM},
        {'role': 'user', 'content': row['prompt']},
    ]


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def sources():
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'scripts').glob('*.py'))}


def train(args, data, tok, manifest):
    encoded = [encode_example(tok, messages(row), row['target']) for row in data]
    for example, row in zip(encoded, data):
        example['target_decision'] = json.loads(row['target'])['decision']
        example['loss_weight'] = class_weight(example['target_decision'], args.class_weighting)
    class_budget = {label: {
        'unique_examples': sum(x['target_decision'] == label for x in encoded),
        'weight_sum': sum(x['loss_weight'] for x in encoded if x['target_decision'] == label),
        'target_tokens': sum(x['target_tokens'] for x in encoded if x['target_decision'] == label),
        'weighted_target_tokens': sum(x['loss_weight'] * x['target_tokens'] for x in encoded if x['target_decision'] == label)
    } for label in ('REPORT', 'CLEAR')}
    effective_batch = args.effective_batch
    steps = args.steps or math.ceil(len(encoded) / effective_batch)
    save_steps = sorted(set(args.save_steps or [steps]))
    if save_steps[-1] != steps or any(s <= 0 or s > steps for s in save_steps):
        raise ValueError('Save steps must be positive and include the final step.')
    order = []
    rng = random.Random(args.seed)
    while len(order) < steps * effective_batch:
        indices = list(range(len(encoded)))
        rng.shuffle(indices)
        order.extend(indices)
    model = load_model(args.adapter, train=True, seed=args.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    total_target = total_prefix = 0
    processed_class_budget = {label: {'examples': 0, 'weight_sum': 0., 'target_tokens': 0, 'weighted_target_tokens': 0.} for label in ('REPORT', 'CLEAR')}
    start = time.monotonic()
    manifest.update(unique_examples=len(data), steps_planned=steps,
                    trainable_parameters=sum(p.numel() for p in parameters),
                    unique_target_tokens=sum(x['target_tokens'] for x in encoded),
                    unique_prefix_tokens=sum(x['prefix_tokens'] for x in encoded),
                    max_sequence_tokens=max(len(x['input_ids']) for x in encoded),
                    supervision='Weighted target-token sum divided by weighted target-token count of the entire effective batch; prefix labels masked',
                    class_weighting=args.class_weighting, unique_class_budget=class_budget,
                    warmup='None; constant predeclared learning rate',
                    sample_order=[data[i]['id'] for i in order[:steps * effective_batch]])
    with (args.output / 'training.jsonl').open('x', buffering=1) as log:
        for step in range(1, steps + 1):
            group = [encoded[i] for i in order[(step - 1) * effective_batch:step * effective_batch]]
            target_count = sum(x['target_tokens'] for x in group)
            denominator = effective_denominator(group)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for index in range(0, len(group), args.batch_size):
                inputs, labels = collate(group[index:index + args.batch_size], tok.pad_token_id)
                losses, _ = token_losses(model, inputs, labels)
                loss = weighted_microbatch_loss(losses, [x['loss_weight'] for x in group[index:index + args.batch_size]], denominator)
                loss.backward()
                loss_value += float(loss.detach())
            before = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0))
            after = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                                  for p in parameters if p.grad is not None))
            if not all(math.isfinite(x) for x in (before, after, loss_value)):
                raise ValueError('Nonfinite training loss/gradient; no completion contract written.')
            if after > 1.0001:
                raise ValueError('Measured clipped gradient exceeds configured norm.')
            optimizer.step()
            for example in group:
                budget = processed_class_budget[example['target_decision']]
                budget['examples'] += 1
                budget['weight_sum'] += example['loss_weight']
                budget['target_tokens'] += example['target_tokens']
                budget['weighted_target_tokens'] += example['loss_weight'] * example['target_tokens']
            total_target += target_count
            total_prefix += sum(x['prefix_tokens'] for x in group)
            row = dict(step=step, loss=loss_value, grad_norm_before_clip=before,
                       grad_norm_after_clip=after, lr=args.lr, target_tokens=total_target,
                       prefix_tokens=total_prefix, weighted_batch_denominator=denominator,
                       processed_class_budget=processed_class_budget, elapsed_s=time.monotonic() - start)
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 8 == 0 or step in save_steps:
                print(json.dumps(row), flush=True)
            if step in save_steps:
                checkpoint = args.output / f'step_{step:04d}'
                model.save_pretrained(checkpoint)
                checkpoint_manifest = dict(manifest, steps=step, finished_at=now(),
                    target_tokens=total_target, prefix_tokens=total_prefix,
                    elapsed_s=time.monotonic() - start,
                    weight_sha256=sha(checkpoint / 'adapter_model.safetensors'),
                    training_log_prefix_sha256=sha(args.output / 'training.jsonl'),
                    training_log_prefix_bytes=(args.output / 'training.jsonl').stat().st_size)
                save_json(checkpoint / 'manifest.json', checkpoint_manifest)
    manifest.update(steps=steps, target_tokens=total_target, prefix_tokens=total_prefix,
                    processed_class_budget=processed_class_budget,
                    elapsed_s=time.monotonic() - start, saved_steps=save_steps,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def decision_scores(model, tok, prompts):
    # Both competing first-divergent tokens are read from the EXACT same logits.
    # This is descriptive conditioning on a canonical JSON prefix, not the primary
    # outcome. All-case unconstrained greedy JSON correctness is the primary measure.
    labels = [tok.encode(word, add_special_tokens=False) for word in ('REPORT', 'CLEAR')]
    if any(len(ids) != 1 for ids in labels):
        raise ValueError('Single-token candidate assumption failed for this tokenizer.')
    canonical = tok.encode('{"decision":"', add_special_tokens=False)
    sequences = [prefix_ids(tok, prompt) + canonical for prompt in prompts]
    width = max(map(len, sequences))
    ids = torch.tensor([[tok.pad_token_id] * (width - len(s)) + s for s in sequences], device='cuda')
    mask = torch.tensor([[0] * (width - len(s)) + [1] * len(s) for s in sequences], device='cuda')
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                   use_cache=False, logits_to_keep=1).logits[:, -1].double().cpu()
    log_probs = logits.log_softmax(-1)
    pair = log_probs[:, [labels[0][0], labels[1][0]]]
    normalized = pair.softmax(-1)
    mass = pair.exp().sum(-1)
    if not bool((mass <= 1 + 1e-12).all()):
        raise ValueError('Incoherent same-logit candidate probability mass.')
    return [dict(p_report=float(normalized[i, 0]), p_clear=float(normalized[i, 1]),
                 candidate_mass=float(mass[i]), logp_report=float(pair[i, 0]),
                 logp_clear=float(pair[i, 1]), canonical_prefix='{\"decision\":\"',
                 candidate_token_ids=[labels[0][0], labels[1][0]]) for i in range(len(prompts))]


def generate(args, data, tok, manifest):
    model = load_model(args.adapter, train=False, seed=args.seed)
    start = time.monotonic()
    with (args.output / 'outputs.jsonl').open('x', buffering=1) as output:
        for begin in range(0, len(data), args.batch_size):
            batch = data[begin:begin + args.batch_size]
            prompts = [messages(row) for row in batch]
            generated = generate_batch(model, tok, prompts, max_new_tokens=args.max_new_tokens,
                                       sample=args.sample, return_details=True)
            scores = decision_scores(model, tok, prompts) if args.score_decision else [None] * len(batch)
            for row, detail, score in zip(batch, generated, scores):
                parsed = parse_output(detail['text'], row)
                result = dict(id=row['id'], source_row_sha256=row['_source_row_sha256'],
                              rendered_row_sha256=sha_row({k: v for k, v in row.items() if k != '_source_row_sha256'}),
                              generated=detail, parsed=parsed)
                if score is not None:
                    result['decision_score'] = score
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
            if begin == 0 or (begin // args.batch_size) % 8 == 0:
                print(json.dumps({'processed': begin + len(batch), 'total': len(data),
                                  'elapsed_s': time.monotonic() - start}), flush=True)
    manifest.update(outputs=len(data), elapsed_s=time.monotonic() - start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    scoring_scope='All-case free generation; canonical-prefix token scoring disabled in this study')
    del model
    gc.collect()
    torch.cuda.empty_cache()


def sha_row(row):
    import hashlib
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('train', 'generate'))
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter')
    p.add_argument('--output-order', choices=('decision_first', 'decision_last'), required=True)
    p.add_argument('--class-weighting', choices=('uniform', 'reweighted'), default='uniform')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--effective-batch', type=int, default=16)
    p.add_argument('--steps', type=int)
    p.add_argument('--save-steps', type=int, nargs='+')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--sample', action='store_true')
    p.add_argument('--score-decision', action='store_true')
    p.add_argument('--max-new-tokens', type=int, default=192)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError(f'Output already exists; preserve it and use a new path: {args.output}')
    args.output.mkdir(parents=True)
    data = read_jsonl(args.data)
    if not data or len({r['id'] for r in data}) != len(data):
        raise ValueError('Dataset must be nonempty with unique IDs.')
    original_data = data
    data = []
    for source_row in original_data:
        if source_row.get('messages'):
            if source_row.get('output_order') != args.output_order:
                raise ValueError('Repair messages must already use the selected serialization.')
            row = dict(source_row)
        else:
            row = serialized_case(source_row, args.output_order)
        row['_source_row_sha256'] = sha_row(source_row)
        data.append(row)
    if args.score_decision:
        raise ValueError('Canonical-prefix decision scoring is disabled in this study.')
    original_sources = sources()
    manifest = dict(started_at=now(), mode=args.mode, args={k: str(v) if isinstance(v, Path) else v
                    for k, v in vars(args).items()}, config=CONFIG,
                    config_sha256=sha(ROOT / 'configs/pilot.json'), source_sha256=original_sources,
                    data_sha256=sha(args.data), source_examples=len(data),
                    initial_adapter_sha256=sha(Path(args.adapter) / 'adapter_model.safetensors') if args.adapter else None)
    save_json(args.output / 'started.json', manifest)
    seed_all(args.seed)
    tok = tokenizer()
    if args.mode == 'train':
        train(args, data, tok, manifest)
    else:
        generate(args, data, tok, manifest)
    if (sources() != original_sources or sha(args.data) != manifest['data_sha256']
            or sha(ROOT / 'configs/pilot.json') != manifest['config_sha256']
            or (args.adapter and sha(Path(args.adapter) / 'adapter_model.safetensors') != manifest['initial_adapter_sha256'])):
        raise ValueError('Source/data drift during model stage; no completion contract written.')
    artifacts = {str(path.relative_to(args.output)): sha(path)
                 for path in sorted(args.output.rglob('*')) if path.is_file()}
    manifest.update(finished_at=now(), status='complete', artifacts_sha256=artifacts)
    save_json(args.output / 'COMPLETED.json', manifest)
    print(json.dumps({'status': 'complete', 'output': str(args.output)}), flush=True)


if __name__ == '__main__':
    main()
