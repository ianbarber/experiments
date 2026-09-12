"""Immutable single-pass training or generation stages for the preliminary check."""
import argparse
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from common import (CONFIG, ROOT, collate, encode_example, generate_batch,
                    generation_settings, load_model, messages, now, prefix_ids,
                    read_jsonl, resolve, row_sampling_seed, schedule_seed, seed_all,
                    sha, sha_row, token_losses, tokenizer)
from task import parse_output


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def sources():
    return {str(path.relative_to(ROOT)): sha(path) for path in sorted((ROOT / 'scripts').glob('*.py'))}


def adapter_identity(adapter):
    if not adapter:
        return None
    directory = resolve(adapter)
    required = ('adapter_model.safetensors', 'adapter_config.json')
    return {name: sha(directory / name) for name in required}


def installed_versions():
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'safetensors', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def check_data(args, data, tok):
    """All feasibility/mask checks run before load_model or seed_all."""
    encoded, token_rows = [], []
    for row in data:
        prompt = messages(row)
        if args.mode == 'train':
            example = encode_example(tok, prompt, row['target'])
            encoded.append(example)
            token_rows.append(dict(id=row['id'], prefix_tokens=example['prefix_tokens'],
                                   target_tokens=example['target_tokens'],
                                   sequence_tokens=len(example['input_ids']),
                                   labels_sha256=sha_row(example['labels']),
                                   input_ids_sha256=sha_row(example['input_ids'])))
        else:
            prefix = prefix_ids(tok, prompt)
            total = len(prefix) + args.max_new_tokens
            if total > CONFIG['max_sequence_length']:
                raise ValueError(f'No silent truncation: {row["id"]} needs {total} reserved tokens.')
            token_rows.append(dict(id=row['id'], prefix_tokens=len(prefix),
                                   reserved_generation_tokens=args.max_new_tokens,
                                   sequence_tokens=total, input_ids_sha256=sha_row(prefix),
                                   sampling_seed=row_sampling_seed(row, args.seed) if args.sample else None))
    with (args.output / 'tokenization.jsonl').open('x') as stream:
        for row in token_rows:
            stream.write(json.dumps(row) + '\n')
    return encoded, dict(examples=len(data), prefix_tokens=sum(x['prefix_tokens'] for x in token_rows),
                         target_tokens=sum(x.get('target_tokens', 0) for x in token_rows),
                         max_sequence_tokens=max(x['sequence_tokens'] for x in token_rows),
                         target_mask_check='passed' if args.mode == 'train' else 'not_applicable',
                         truncation='none')


def parameter_fingerprint(parameters):
    """Fingerprint trainable values independently of serialized checkpoint bytes."""
    import hashlib
    digest = hashlib.sha256()
    for name, parameter in parameters:
        values = parameter.detach().float().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def train(args, data, tok, encoded, manifest):
    import torch
    order_seed = schedule_seed(args.seed, 'training_pass_order', args.epoch)
    order = list(range(len(encoded)))
    random.Random(order_seed).shuffle(order)
    steps = math.ceil(len(encoded) / args.effective_batch)
    model = load_model(args.adapter, train=True, seed=args.seed)
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not named_parameters or any('lora_' not in name for name, _ in named_parameters):
        raise ValueError('Expected only LoRA adapter parameters to be trainable.')
    parameters = [parameter for _, parameter in named_parameters]
    initial_values = parameter_fingerprint(named_parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    continuation = dict(seed=args.seed, data_sha256=manifest['data_sha256'],
                        config_sha256=manifest['config_sha256'], lr=args.lr,
                        effective_batch=args.effective_batch,
                        trainable_parameter_names=[name for name, _ in named_parameters])
    previous_steps = 0
    if args.optimizer:
        saved = torch.load(resolve(args.optimizer), map_location='cpu', weights_only=True)
        for key, value in continuation.items():
            if saved['continuation'][key] != value:
                raise ValueError(f'Optimizer continuity mismatch: {key}')
        if saved['epoch'] != args.epoch - 1:
            raise ValueError('Optimizer does not belong to the immediately preceding pass.')
        if saved['adapter_identity'] != manifest['initial_adapter_identity']:
            raise ValueError('Optimizer is not paired with the exact input adapter.')
        if saved['trainable_parameter_sha256'] != initial_values:
            raise ValueError('Reloaded adapter values differ from the saved optimizer checkpoint.')
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        previous_steps = saved['cumulative_steps']
        if any(float(group['lr']) != args.lr for group in optimizer.param_groups):
            raise ValueError('Resumed optimizer changed the declared learning rate.')
    # Each pass has a fixed RNG stream, independent of model loading/init draws.
    # Dropout is zero; optimizer moments continue across selected adjacent passes.
    training_seed = schedule_seed(args.seed, 'training_pass_rng', args.epoch)
    seed_all(training_seed)
    total_target = total_prefix = 0
    start = time.monotonic()
    manifest.update(unique_examples=len(data), steps_planned=steps, epoch=args.epoch,
                    order_seed=order_seed, training_rng_seed=training_seed,
                    optimizer_resumed=bool(args.optimizer), optimizer_previous_steps=previous_steps,
                    trainable_parameters=sum(parameter.numel() for parameter in parameters),
                    initial_trainable_parameter_sha256=initial_values,
                    trainable_parameter_names=[name for name, _ in named_parameters],
                    supervision='Sum of target-token losses divided by target tokens in the entire effective batch; prefix labels masked, activations not detached',
                    warmup='None; constant learning rate',
                    sample_order=[data[index]['id'] for index in order])
    with (args.output / 'training.jsonl').open('x', buffering=1) as log:
        for step, begin in enumerate(range(0, len(order), args.effective_batch), start=1):
            indices = order[begin:begin + args.effective_batch]
            group = [encoded[index] for index in indices]
            denominator = sum(example['target_tokens'] for example in group)
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for index in range(0, len(group), args.batch_size):
                inputs, labels = collate(group[index:index + args.batch_size], tok.pad_token_id)
                losses, active = token_losses(model, inputs, labels)
                if int(active.sum()) != sum(x['target_tokens'] for x in group[index:index + args.batch_size]):
                    raise ValueError('Collated target count differs from the frozen token budget.')
                loss = losses.sum() / denominator
                loss.backward()
                loss_value += float(loss.detach())
            before = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            after = math.sqrt(sum(float(parameter.grad.detach().float().square().sum())
                                  for parameter in parameters if parameter.grad is not None))
            if not all(math.isfinite(value) for value in (before, after, loss_value)):
                raise ValueError('Nonfinite training loss or gradient.')
            if after > 1.0001:
                raise ValueError('Measured clipped gradient exceeds norm 1.')
            optimizer.step()
            total_target += denominator
            total_prefix += sum(example['prefix_tokens'] for example in group)
            row = dict(step=step, cumulative_step=previous_steps + step, epoch=args.epoch,
                       loss=loss_value, grad_norm_before_clip=before, grad_norm_after_clip=after,
                       lr=args.lr, examples=len(indices), ids=[data[index]['id'] for index in indices],
                       effective_batch_target_tokens=denominator,
                       target_tokens=total_target, prefix_tokens=total_prefix,
                       elapsed_s=time.monotonic() - start)
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 8 == 0 or step == steps:
                print(json.dumps(row), flush=True)
    if total_target != sum(example['target_tokens'] for example in encoded):
        raise ValueError('This invocation did not process exactly one pass of target tokens.')
    final_values = parameter_fingerprint(named_parameters)
    if final_values == initial_values:
        raise ValueError('Training made no change to adapter parameters.')
    checkpoint = args.output / 'adapter'
    model.save_pretrained(checkpoint)
    identity = adapter_identity(checkpoint)
    torch.save(dict(optimizer_state_dict=optimizer.state_dict(), continuation=continuation,
                    epoch=args.epoch, cumulative_steps=previous_steps + steps,
                    adapter_identity=identity, trainable_parameter_sha256=final_values),
               args.output / 'optimizer.pt')
    manifest.update(steps=steps, cumulative_optimizer_steps=previous_steps + steps,
                    target_tokens=total_target, prefix_tokens=total_prefix,
                    adapter_identity=identity, final_trainable_parameter_sha256=final_values,
                    elapsed_s=time.monotonic() - start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    save_json(checkpoint / 'manifest.json', dict(manifest, finished_at=now()))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


def generate(args, data, tok, manifest):
    import torch
    model = load_model(args.adapter, train=False, seed=args.seed)
    start = time.monotonic()
    total_tokens = 0
    # Independent, reset per-row RNG prevents compaction or row ordering from
    # changing sampled collection traces. Greedy evaluation can safely batch.
    batch_size = 1 if args.sample else args.batch_size
    with (args.output / 'outputs.jsonl').open('x', buffering=1) as output:
        for begin in range(0, len(data), batch_size):
            batch = data[begin:begin + batch_size]
            sampling_seed = row_sampling_seed(batch[0], args.seed) if args.sample else None
            if args.sample:
                seed_all(sampling_seed)
            generated = generate_batch(model, tok, [messages(row) for row in batch],
                                       max_new_tokens=args.max_new_tokens, sample=args.sample)
            if len(generated) != len(batch):
                raise ValueError('Generation did not return exactly one response per input.')
            for row, detail in zip(batch, generated):
                parsed = parse_output(detail['text'], finish_reason=detail['finish_reason'])
                if detail['unexpected_special_token_ids']:
                    parsed = dict(valid=False, decision=None, reason='', error='unexpected_special_token')
                result = dict(id=row['id'], source_id=row.get('source_id', row.get('case_id', row['id'])),
                              draw_index=row.get('draw_index'), source_row_sha256=sha_row(row),
                              gold_decision=row.get('gold_decision'), generated=detail, parsed=parsed,
                              sampling_seed=sampling_seed,
                              adapter_sha256=(manifest['initial_adapter_identity'] or {}).get('adapter_model.safetensors'))
                total_tokens += detail['generated_tokens']
                output.write(json.dumps(result, ensure_ascii=False) + '\n')
            if begin == 0 or (begin // batch_size) % 16 == 0 or begin + len(batch) == len(data):
                print(json.dumps(dict(processed=begin + len(batch), total=len(data),
                                      generated_tokens=total_tokens, elapsed_s=time.monotonic() - start)), flush=True)
    manifest.update(outputs=len(data), generated_tokens=total_tokens,
                    effective_generation_batch_size=batch_size, elapsed_s=time.monotonic() - start,
                    generation_settings=generation_settings(tok, args.sample, args.max_new_tokens),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    scoring_scope='All-case free generation; no canonical-prefix scoring')
    del model
    gc.collect()
    torch.cuda.empty_cache()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'generate'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--optimizer', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--effective-batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, choices=(1, 2), default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1 or args.effective_batch < 1 or args.batch_size > args.effective_batch:
        parser.error('Require 1 <= batch-size <= effective-batch.')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Require a finite positive learning rate and a 32-bit nonnegative seed.')
    if args.max_new_tokens != 192:
        parser.error('The frozen preliminary-check generation cap is 192.')
    if args.mode == 'train':
        if args.sample:
            parser.error('--sample is only valid for generation.')
        if args.epoch == 2 and (not args.optimizer or not args.adapter):
            parser.error('Pass 2 must continue both the previous adapter and optimizer.')
        if args.epoch == 1 and args.optimizer:
            parser.error('Pass 1 starts a fresh optimizer, including induction from a competent adapter.')
    elif args.optimizer:
        parser.error('--optimizer is only valid for training.')
    for name in ('data', 'output', 'adapter', 'optimizer'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, resolve(value))
    return args


def main():
    args = arguments()
    if args.output.exists():
        raise ValueError(f'Output already exists; preserve it and use a new path: {args.output}')
    data = read_jsonl(args.data)
    if not data or len({row['id'] for row in data}) != len(data):
        raise ValueError('Dataset must be nonempty with unique IDs.')
    source_hashes = sources()
    manifest = dict(started_at=now(), mode=args.mode, check_only=args.check_only,
                    args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    config=CONFIG, config_sha256=sha(ROOT / 'configs/pilot.json'),
                    source_sha256=source_hashes, data_sha256=sha(args.data), source_examples=len(data),
                    initial_adapter_identity=adapter_identity(args.adapter),
                    initial_optimizer_sha256=sha(args.optimizer) if args.optimizer else None,
                    package_versions=installed_versions())
    args.output.mkdir(parents=True)
    save_json(args.output / 'started.json', manifest)
    start = time.monotonic()
    try:
        tok = tokenizer()
        encoded, token_summary = check_data(args, data, tok)
        manifest['tokenization'] = token_summary
        if not args.check_only:
            if args.mode == 'train':
                train(args, data, tok, encoded, manifest)
            else:
                generate(args, data, tok, manifest)
        if (sources() != source_hashes or sha(args.data) != manifest['data_sha256']
                or sha(ROOT / 'configs/pilot.json') != manifest['config_sha256']
                or adapter_identity(args.adapter) != manifest['initial_adapter_identity']
                or (args.optimizer and sha(args.optimizer) != manifest['initial_optimizer_sha256'])):
            raise ValueError('Source/data/config/input-checkpoint drift during stage; completion withheld.')
        artifacts = {str(path.relative_to(args.output)): sha(path)
                     for path in sorted(args.output.rglob('*')) if path.is_file()}
        manifest.update(finished_at=now(), status='complete', stage_elapsed_s=time.monotonic() - start,
                        artifacts_sha256=artifacts)
        save_json(args.output / 'COMPLETED.json', manifest)
        print(json.dumps(dict(status='complete', check_only=args.check_only, output=str(args.output))), flush=True)
    except Exception as error:
        save_json(args.output / 'FAILED.json', dict(started_at=manifest['started_at'], failed_at=now(),
                                                   error_type=type(error).__name__, error=str(error),
                                                   stage_elapsed_s=time.monotonic() - start))
        raise


if __name__ == '__main__':
    main()
