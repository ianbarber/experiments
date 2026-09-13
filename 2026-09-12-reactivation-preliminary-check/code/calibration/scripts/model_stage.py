"""Immutable calibration training, combined diagnostics, and frozen-batch generation."""
import argparse
from collections import Counter
import gc
import importlib.metadata
import json
import math
from pathlib import Path
import random
import time

from common import (CONFIG, ROOT, collate, encode_example, generate_batch,
                    generation_settings, load_model, messages, now, prefix_ids,
                    read_jsonl, resolve, sampling_batch_metadata, schedule_seed, seed_all,
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
    return {name: sha(directory / name) for name in ('adapter_model.safetensors', 'adapter_config.json')}


def installed_versions():
    versions = {}
    for name in ('torch', 'transformers', 'peft', 'bitsandbytes', 'tokenizers', 'safetensors', 'numpy'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def group_names(row):
    if type(row.get('is_bad')) is not bool or row.get('gold_decision') not in ('REPORT', 'CLEAR'):
        raise ValueError('Loss rows require Boolean is_bad and a gold REPORT/CLEAR decision.')
    if row['is_bad']:
        if row['gold_decision'] != 'REPORT':
            raise ValueError('Authored undesirable targets must be REPORT cases.')
        category = row.get('error_category', row.get('authored_error_category'))
        if not isinstance(category, str) or not category:
            raise ValueError('Each bad target must identify its authored error category.')
        return ('all', 'bad', 'bad_category:' + category)
    return ('all', 'good', 'good_' + row['gold_decision'].lower())


def loss_weight(row, bad_weight):
    group_names(row)
    return bad_weight if row['is_bad'] else 1.0


def validate_pool(data, bad_only=False):
    groups = Counter('bad' if row['is_bad'] else 'good_' + row['gold_decision'].lower() for row in data)
    expected = {'bad': 512} if bad_only else {'bad': 512, 'good_report': 256, 'good_clear': 256}
    if groups != expected:
        if bad_only:
            raise ValueError('The acquisition diagnostic requires exactly512 all-is_bad rows.')
        raise ValueError('Each calibration training/loss pool must have 512 bad, 256 good REPORT, and 256 good CLEAR rows.')
    for row in data:
        group_names(row)
        expected_decision = 'CLEAR' if row['is_bad'] else row['gold_decision']
        parsed = parse_output(row['target'], finish_reason='eos')
        if (not parsed['valid'] or parsed['decision'] != expected_decision
                or row.get('target_decision', expected_decision) != expected_decision):
            raise ValueError('Authored target decision disagrees with is_bad/gold metadata.')


def stratified_order(data, seed, epoch, effective_batch=16, bad_only=False):
    """One exact pass, 8 bad + 4 good REPORT + 4 good CLEAR per effective batch."""
    if effective_batch != 16:
        raise ValueError('The frozen stratified effective batch is 16.')
    if bad_only:
        if not data or len(data) % 16 or any(row.get('is_bad') is not True for row in data):
            raise ValueError('Pure-bad acquisition batches require only bad rows and complete batches.')
        indices = list(range(len(data)))
        random.Random(schedule_seed(seed, 'pure_bad_diagnostic_pass', epoch)).shuffle(indices)
        return indices
    groups = {name: [] for name in ('bad', 'good_report', 'good_clear')}
    for index, row in enumerate(data):
        group_names(row)
        groups['bad' if row['is_bad'] else 'good_' + row['gold_decision'].lower()].append(index)
    count = len(data) // 16
    if len(data) != count * 16 or any(len(groups[name]) != count * quota
            for name, quota in (('bad', 8), ('good_report', 4), ('good_clear', 4))):
        raise ValueError('Stratified ordering requires complete 8:4:4 batches with no repetitions.')
    for name, indices in groups.items():
        random.Random(schedule_seed(seed, 'stratified_pass_group', {'epoch': epoch, 'group': name})).shuffle(indices)
    order = []
    for batch in range(count):
        indices = groups['bad'][batch * 8:(batch + 1) * 8]
        indices += groups['good_report'][batch * 4:(batch + 1) * 4]
        indices += groups['good_clear'][batch * 4:(batch + 1) * 4]
        random.Random(schedule_seed(seed, 'stratified_within_batch', {'epoch': epoch, 'batch': batch})).shuffle(indices)
        order.extend(indices)
    if sorted(order) != list(range(len(data))):
        raise ValueError('Stratification changed exact example coverage.')
    return order


def weighted_loss(losses, weights, effective_weighted_tokens):
    import torch
    if not math.isfinite(effective_weighted_tokens) or effective_weighted_tokens <= 0:
        raise ValueError('Effective weighted target-token denominator must be positive and finite.')
    row_weights = torch.tensor(weights, device=losses.device, dtype=losses.dtype)
    return (losses.sum(-1) * row_weights).sum() / effective_weighted_tokens


def add_groups(summary, row, loss_sum, token_count, weight):
    if not math.isfinite(loss_sum) or token_count <= 0:
        raise ValueError('Invalid per-row target loss.')
    for name in group_names(row):
        result = summary.setdefault(name, dict(examples=0, target_tokens=0, loss_sum=0.0,
                                               weighted_loss_sum=0.0, weighted_target_tokens=0.0))
        result['examples'] += 1
        result['target_tokens'] += token_count
        result['loss_sum'] += loss_sum
        result['weighted_loss_sum'] += loss_sum * weight
        result['weighted_target_tokens'] += token_count * weight


def finish_groups(summary):
    return {name: dict(value, mean_target_nll=value['loss_sum'] / value['target_tokens'],
                       weighted_mean_target_nll=value['weighted_loss_sum'] / value['weighted_target_tokens'])
            for name, value in summary.items()}


def check_data(args, data, generation_sets, tok):
    encoded, token_rows = [], []
    if args.mode in ('train', 'diagnose'):
        validate_pool(data, args.bad_only_diagnostic)
        for row in data:
            example = encode_example(tok, messages(row), row['target'])
            encoded.append(example)
            token_rows.append(dict(cohort='target_loss', id=row['id'], prefix_tokens=example['prefix_tokens'],
                                   target_tokens=example['target_tokens'], sequence_tokens=len(example['input_ids']),
                                   labels_sha256=sha_row(example['labels']), input_ids_sha256=sha_row(example['input_ids']),
                                   is_bad=row['is_bad'], loss_weight=loss_weight(row, args.bad_weight)))
    for name, rows in generation_sets:
        for row in rows:
            prefix = prefix_ids(tok, messages(row))
            total = len(prefix) + args.max_new_tokens
            if total > CONFIG['max_sequence_length']:
                raise ValueError(f'No silent truncation: {name}/{row["id"]} needs {total} reserved tokens.')
            token_rows.append(dict(cohort=name, id=row['id'], prefix_tokens=len(prefix),
                                   reserved_generation_tokens=args.max_new_tokens,
                                   sequence_tokens=total, input_ids_sha256=sha_row(prefix)))
    with (args.output / 'tokenization.jsonl').open('x') as stream:
        for row in token_rows:
            stream.write(json.dumps(row) + '\n')
    return encoded, dict(tokenized_examples=len(token_rows), target_examples=len(encoded),
                         target_tokens=sum(row.get('target_tokens', 0) for row in token_rows),
                         max_sequence_tokens=max(row['sequence_tokens'] for row in token_rows), truncation='none')


def parameter_fingerprint(parameters):
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
    order = stratified_order(data, args.seed, args.epoch, args.effective_batch, args.bad_only_diagnostic)
    steps = len(order) // args.effective_batch
    model = load_model(args.adapter, train=True, seed=args.seed)
    named_parameters = [(name, value) for name, value in model.named_parameters() if value.requires_grad]
    if not named_parameters or any('lora_' not in name for name, _ in named_parameters):
        raise ValueError('Expected only LoRA adapter parameters to be trainable.')
    parameters = [value for _, value in named_parameters]
    initial_values = parameter_fingerprint(named_parameters)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.0)
    continuation = dict(seed=args.seed, data_sha256=manifest['data_sha256'], config_sha256=manifest['config_sha256'],
                        lr=args.lr, effective_batch=args.effective_batch, bad_weight=args.bad_weight,
                        rule_variant=args.rule_variant, bad_only_diagnostic=args.bad_only_diagnostic,
                        trainable_parameter_names=[name for name, _ in named_parameters])
    previous_steps = 0
    if args.optimizer:
        saved = torch.load(resolve(args.optimizer), map_location='cpu', weights_only=True)
        if saved['continuation'] != continuation or saved['epoch'] != args.epoch - 1:
            raise ValueError('Optimizer continuity mismatch: recipe/data/config/seed/epoch changed.')
        if saved['adapter_identity'] != manifest['initial_adapter_identity'] or saved['trainable_parameter_sha256'] != initial_values:
            raise ValueError('Optimizer is not paired with the exact input adapter weights/configuration.')
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        previous_steps = saved['cumulative_steps']
        if any(float(group['lr']) != args.lr for group in optimizer.param_groups):
            raise ValueError('Resumed optimizer changed the declared learning rate.')
    training_seed = schedule_seed(args.seed, 'training_pass_rng', args.epoch)
    seed_all(training_seed)
    cumulative_groups = {}
    total_prefix = 0
    start = time.monotonic()
    exposure = [dict(id=data[index]['id'], source_row_sha256=sha_row(data[index]),
                     target_tokens=encoded[index]['target_tokens'], weight=loss_weight(data[index], args.bad_weight)) for index in order]
    manifest.update(unique_examples=len(data), steps_planned=steps, epoch=args.epoch,
                    training_rng_seed=training_seed, optimizer_resumed=bool(args.optimizer), optimizer_previous_steps=previous_steps,
                    bad_weight=args.bad_weight, rule_variant=args.rule_variant, bad_only_diagnostic=args.bad_only_diagnostic,
                    trainable_parameters=sum(value.numel() for value in parameters),
                    initial_trainable_parameter_sha256=initial_values,
                    trainable_parameter_names=[name for name, _ in named_parameters],
                    supervision='Weighted target-token loss sum / weighted target tokens of the entire effective batch; prefix masked, not detached',
                    stratification=('Pure-bad acquisition diagnostic; fixed shuffled one-pass coverage' if args.bad_only_diagnostic
                                    else 'Every effective batch:8 bad,4 good REPORT,4 good CLEAR; exact one-pass coverage'),
                    warmup='None; constant learning rate', sample_order=[data[index]['id'] for index in order],
                    actual_exposure_sha256=sha_row(exposure))
    with (args.output / 'exposure.jsonl').open('x') as stream:
        for row in exposure:
            stream.write(json.dumps(row) + '\n')
    with (args.output / 'training.jsonl').open('x', buffering=1) as log:
        for step, begin in enumerate(range(0, len(order), args.effective_batch), start=1):
            indices = order[begin:begin + args.effective_batch]
            weights = [loss_weight(data[index], args.bad_weight) for index in indices]
            denominator = sum(encoded[index]['target_tokens'] * weight for index, weight in zip(indices, weights))
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            batch_groups = {}
            for offset in range(0, len(indices), args.batch_size):
                micro_indices = indices[offset:offset + args.batch_size]
                inputs, labels = collate([encoded[index] for index in micro_indices], tok.pad_token_id)
                losses, active = token_losses(model, inputs, labels)
                if int(active.sum()) != sum(encoded[index]['target_tokens'] for index in micro_indices):
                    raise ValueError('Collated target count differs from the frozen token budget.')
                micro_weights = weights[offset:offset + args.batch_size]
                loss = weighted_loss(losses, micro_weights, denominator)
                loss.backward()
                loss_value += float(loss.detach())
                sums = losses.detach().sum(-1).cpu().tolist()
                for index, row_sum, weight in zip(micro_indices, sums, micro_weights):
                    add_groups(batch_groups, data[index], row_sum, encoded[index]['target_tokens'], weight)
                    add_groups(cumulative_groups, data[index], row_sum, encoded[index]['target_tokens'], weight)
            before = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
            after = math.sqrt(sum(float(value.grad.detach().float().square().sum()) for value in parameters if value.grad is not None))
            if not all(math.isfinite(value) for value in (before, after, loss_value)) or after > 1.0001:
                raise ValueError('Nonfinite loss/gradient or clipped gradient above norm1.')
            optimizer.step()
            total_prefix += sum(encoded[index]['prefix_tokens'] for index in indices)
            row = dict(step=step, cumulative_step=previous_steps + step, epoch=args.epoch,
                       loss=loss_value, grad_norm_before_clip=before, grad_norm_after_clip=after, lr=args.lr,
                       ids=[data[index]['id'] for index in indices], weighted_target_token_denominator=denominator,
                       groups_at_processing=finish_groups(batch_groups),
                       cumulative_groups_at_processing=finish_groups(cumulative_groups), elapsed_s=time.monotonic() - start)
            log.write(json.dumps(row) + '\n')
            if step == 1 or step % 8 == 0 or step == steps:
                print(json.dumps({key: row[key] for key in ('step', 'loss', 'grad_norm_before_clip', 'elapsed_s')}), flush=True)
    if cumulative_groups['all']['target_tokens'] != sum(example['target_tokens'] for example in encoded):
        raise ValueError('This invocation did not process exactly one pass of target tokens.')
    final_values = parameter_fingerprint(named_parameters)
    if final_values == initial_values:
        raise ValueError('Training made no adapter parameter change.')
    checkpoint = args.output / 'adapter'
    model.save_pretrained(checkpoint)
    identity = adapter_identity(checkpoint)
    torch.save(dict(optimizer_state_dict=optimizer.state_dict(), continuation=continuation, epoch=args.epoch,
                    cumulative_steps=previous_steps + steps, adapter_identity=identity,
                    trainable_parameter_sha256=final_values), args.output / 'optimizer.pt')
    manifest.update(steps=steps, cumulative_optimizer_steps=previous_steps + steps,
                    target_tokens=cumulative_groups['all']['target_tokens'], prefix_tokens=total_prefix,
                    groups_at_processing=finish_groups(cumulative_groups), adapter_identity=identity,
                    final_trainable_parameter_sha256=final_values, elapsed_s=time.monotonic() - start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
    save_json(checkpoint / 'manifest.json', dict(manifest, finished_at=now()))
    del optimizer, model
    gc.collect()
    torch.cuda.empty_cache()


def generate_sets(args, model, tok, generation_sets, manifest):
    total_tokens = total_outputs = 0
    start = time.monotonic()
    batch_size = CONFIG.get('sampling_batch_size', 16) if args.sample else args.batch_size
    identities = []
    with (args.output / 'outputs.jsonl').open('x', buffering=1) as output:
        for cohort, data in generation_sets:
            for begin in range(0, len(data), batch_size):
                batch = data[begin:begin + batch_size]
                sampling = sampling_batch_metadata(batch, args.seed, begin // batch_size) if args.sample else None
                if sampling:
                    seed_all(sampling['batch_seed'])
                generated = generate_batch(model, tok, [messages(row) for row in batch], max_new_tokens=args.max_new_tokens, sample=args.sample)
                if len(generated) != len(batch):
                    raise ValueError('Generation did not return exactly one response per input.')
                identities.append(dict(cohort=cohort, ids=[row['id'] for row in batch], sampling=sampling))
                for row, detail in zip(batch, generated):
                    parsed = parse_output(detail['text'], finish_reason=detail['finish_reason'])
                    if detail['unexpected_special_token_ids']:
                        parsed = dict(valid=False, decision=None, reason='', error='unexpected_special_token')
                    record = dict(id=row['id'], cohort=cohort, source_file=manifest['generation_files'][cohort],
                                  source_id=row.get('source_id', row.get('case_id', row['id'])),
                                  draw_index=row.get('draw_index'), source_row_sha256=sha_row(row),
                                  gold_decision=row.get('gold_decision'), is_bad=row.get('is_bad'), generated=detail,
                                  parsed=parsed, sampling=sampling,
                                  adapter_sha256=(manifest['initial_adapter_identity'] or {}).get('adapter_model.safetensors'))
                    output.write(json.dumps(record, ensure_ascii=False) + '\n')
                    total_tokens += detail['generated_tokens']
                    total_outputs += 1
                if begin == 0 or (begin // batch_size) % 8 == 0 or begin + len(batch) == len(data):
                    print(json.dumps(dict(cohort=cohort, processed=begin + len(batch), total=len(data),
                                          generated_tokens=total_tokens, elapsed_s=time.monotonic() - start)), flush=True)
    save_json(args.output / 'generation_batches.json', identities)
    return dict(outputs=total_outputs, generated_tokens=total_tokens, generation_elapsed_s=time.monotonic() - start,
                effective_generation_batch_size=batch_size, generation_batch_inventory_sha256=sha_row(identities),
                generation_settings=generation_settings(tok, args.sample, args.max_new_tokens))


def target_loss_diagnostics(args, model, tok, data, encoded):
    import torch
    start = time.monotonic()
    groups = {}
    with (args.output / 'target_losses.jsonl').open('x', buffering=1) as stream, torch.inference_mode():
        for begin in range(0, len(data), args.batch_size):
            batch_rows = data[begin:begin + args.batch_size]
            batch_encoded = encoded[begin:begin + args.batch_size]
            inputs, labels = collate(batch_encoded, tok.pad_token_id)
            losses, active = token_losses(model, inputs, labels)
            sums, counts = losses.sum(-1).cpu().tolist(), active.sum(-1).cpu().tolist()
            for row, example, loss_sum, count in zip(batch_rows, batch_encoded, sums, counts):
                if count != example['target_tokens']:
                    raise ValueError('Teacher-forced diagnostic target count changed.')
                weight = loss_weight(row, args.bad_weight)
                add_groups(groups, row, loss_sum, count, weight)
                stream.write(json.dumps(dict(id=row['id'], source_row_sha256=sha_row(row), groups=group_names(row),
                                              target_tokens=count, target_loss_sum=loss_sum, mean_target_nll=loss_sum / count,
                                              loss_weight=weight), ensure_ascii=False) + '\n')
    summary = dict(groups=finish_groups(groups), examples=len(data), elapsed_s=time.monotonic() - start,
                   interpretation='Teacher-forced target likelihood, including EOS; not free-generation correctness or evidence of hidden reasoning')
    save_json(args.output / 'target_loss_summary.json', summary)
    return summary


def inference(args, data, tok, encoded, generation_sets, manifest):
    import torch
    model = load_model(args.adapter, train=False, seed=args.seed)
    named_parameters = [(name, value) for name, value in model.named_parameters() if 'lora_' in name]
    if not named_parameters:
        raise ValueError('Expected a loaded LoRA adapter for read-only calibration diagnostics.')
    initial = parameter_fingerprint(named_parameters)
    start = time.monotonic()
    if not args.loss_only:
        manifest.update(generate_sets(args, model, tok, generation_sets, manifest))
    if args.mode == 'diagnose':
        manifest['target_loss_diagnostics'] = target_loss_diagnostics(args, model, tok, data, encoded)
    final = parameter_fingerprint(named_parameters)
    if initial != final:
        raise ValueError('Read-only diagnostic changed model parameters.')
    manifest.update(inference_parameter_sha256=initial, elapsed_s=time.monotonic() - start,
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                    scoring_scope='All-case free generation plus separately labeled teacher-forced target diagnostics')
    del model
    gc.collect()
    torch.cuda.empty_cache()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'generate', 'diagnose'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--generation-data', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--optimizer', type=Path)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--effective-batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bad-weight', type=float, choices=(1 / 3, 1.0, 3.0), default=1.0)
    parser.add_argument('--rule-variant', choices=('present', 'omitted'), required=True)
    parser.add_argument('--bad-only-diagnostic', action='store_true')
    parser.add_argument('--loss-only', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--max-new-tokens', type=int, default=192)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if args.batch_size is None:
        args.batch_size = CONFIG['micro_batch_size'] if args.mode == 'train' else CONFIG['generation_batch_size']
    if args.batch_size < 1 or args.effective_batch != 16:
        parser.error('Positive batch size and effective batch16 are required.')
    if args.mode == 'train' and args.batch_size > 16:
        parser.error('Training microbatch cannot exceed the effective batch16.')
    if not math.isfinite(args.lr) or args.lr <= 0 or not 0 <= args.seed < 2**32:
        parser.error('Require finite positive learning rate and nonnegative32-bit seed.')
    if args.max_new_tokens != 192:
        parser.error('The frozen generation cap is192.')
    if args.mode == 'train':
        if args.sample or args.generation_data:
            parser.error('Training accepts neither sampling nor generation-data.')
        if args.epoch > 1 and (not args.optimizer or not args.adapter):
            parser.error('Passes2–4 must continue both previous adapter and optimizer.')
        if args.epoch == 1 and args.optimizer:
            parser.error('Pass1 starts a fresh optimizer from the competent adapter.')
    elif args.optimizer:
        parser.error('Optimizer is only valid for training.')
    if args.mode == 'diagnose' and ((not args.generation_data and not args.loss_only) or args.sample):
        parser.error('diagnose requires ordered generation-data and uses greedy decoding.')
    if args.loss_only and (args.mode != 'diagnose' or args.generation_data):
        parser.error('--loss-only is diagnose-only and must omit generation-data.')
    if args.mode == 'generate' and args.generation_data:
        parser.error('Standalone generate uses --data, not generation-data.')
    if not args.adapter and not args.check_only:
        parser.error('Actual calibration stages require an explicit starting adapter.')
    for name in ('data', 'output', 'adapter', 'optimizer'):
        if getattr(args, name) is not None:
            setattr(args, name, resolve(getattr(args, name)))
    if args.generation_data:
        args.generation_data = [resolve(path) for path in args.generation_data]
    return args


def checked_rows(path):
    rows = read_jsonl(path)
    if not rows or len({row['id'] for row in rows}) != len(rows):
        raise ValueError(f'Dataset must be nonempty with unique IDs: {path}')
    return rows


def main():
    args = arguments()
    if args.output.exists():
        raise ValueError(f'Output exists; preserve it and use a new path: {args.output}')
    data = checked_rows(args.data)
    generation_sets = ([(path.stem, checked_rows(path)) for path in (args.generation_data or [])] if args.mode == 'diagnose'
                       else [(args.data.stem, data)] if args.mode == 'generate' else [])
    if len({name for name, _ in generation_sets}) != len(generation_sets):
        raise ValueError('Generation cohort file stems must be unique.')
    original_sources = sources()
    paths = [args.data] + (args.generation_data or [])
    input_hashes = {str(path): sha(path) for path in paths}
    manifest = dict(started_at=now(), mode=args.mode, check_only=args.check_only,
                    args={key: str(value) if isinstance(value, Path) else [str(path) for path in value]
                          if key == 'generation_data' and value is not None else value for key, value in vars(args).items()},
                    config=CONFIG, config_sha256=sha(ROOT / 'configs/pilot.json'), source_sha256=original_sources,
                    generation_files={path.stem: str(path.relative_to(ROOT)) for path in
                                      (args.generation_data or ([args.data] if args.mode == 'generate' else []))},
                    input_files_sha256=input_hashes, data_sha256=sha(args.data), source_examples=len(data),
                    initial_adapter_identity=adapter_identity(args.adapter),
                    initial_optimizer_sha256=sha(args.optimizer) if args.optimizer else None, package_versions=installed_versions())
    args.output.mkdir(parents=True)
    save_json(args.output / 'started.json', manifest)
    start = time.monotonic()
    try:
        tok = tokenizer()
        encoded, token_summary = check_data(args, data, generation_sets, tok)
        manifest['tokenization'] = token_summary
        if not args.check_only:
            if args.mode == 'train':
                train(args, data, tok, encoded, manifest)
            else:
                inference(args, data, tok, encoded, generation_sets, manifest)
        if (sources() != original_sources or any(sha(path) != digest for path, digest in input_hashes.items())
                or sha(ROOT / 'configs/pilot.json') != manifest['config_sha256']
                or adapter_identity(args.adapter) != manifest['initial_adapter_identity']
                or (args.optimizer and sha(args.optimizer) != manifest['initial_optimizer_sha256'])):
            raise ValueError('Source/data/config/input-checkpoint drift during stage; completion withheld.')
        manifest.update(finished_at=now(), status='complete', stage_elapsed_s=time.monotonic() - start,
                        artifacts_sha256={str(path.relative_to(args.output)): sha(path)
                                          for path in sorted(args.output.rglob('*')) if path.is_file()})
        save_json(args.output / 'COMPLETED.json', manifest)
        print(json.dumps(dict(status='complete', check_only=args.check_only, output=str(args.output))), flush=True)
    except Exception as error:
        save_json(args.output / 'FAILED.json', dict(started_at=manifest['started_at'], failed_at=now(),
                                                   error_type=type(error).__name__, error=str(error),
                                                   stage_elapsed_s=time.monotonic() - start))
        raise


if __name__ == '__main__':
    main()
