# Additional exact source excerpts

These slices supplement the original, unchanged support directory. Line bounds
refer to the original frozen files. Full-file and excerpt hashes are recorded
below and in the publication packet manifest; excerpts do not reconstruct a
complete source file.

## new_training_loop

Original: `scripts/model_stage.py`, lines 37–125, SHA256 `026a07baa55eb76a6d90bc8b21178cdb6d19fc4b07a8ceac6267c22d4f709816`.
Excerpt SHA256: `bcc10c974abaf9009be394a0718e5088bb918f420eca7ff888bada549d695b9e`.

```python
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
```

## new_split_generation

Original: `scripts/make_data.py`, lines 266–302, SHA256 `d2ccf9cf996496204bbe726bf3a88ae2ad3e84e0a49e052cf7d98b18ee8cb252`.
Excerpt SHA256: `366374543deb36ecd82f0728ff4ae5c9975b3641a1574a51ff62024ec9276cc6`.

```python
def split_seeds(seed: int) -> dict[str, int]:
    if type(seed) is not int or seed < 0:
        raise ValueError("Seed must be a nonnegative integer")
    return {name: seed + offset for offset, name in enumerate(SPLIT_SIZES)}


def build_evaluation(seed: int, seen: set[str]) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for suite, n, domains, style in (("id", 64, TRAIN_DOMAINS, "table"),
                                    ("heldout", 64, HELDOUT_DOMAINS, "table"),
                                    ("narrative", 128, TRAIN_DOMAINS + HELDOUT_DOMAINS, "narrative")):
        for stratum in ("trigger_report", "control_report", "clear"):
            added = build_rows("evaluation", n, stratum, rng, seen, domains=domains,
                               style=style, offset=len(rows))
            for row in added:
                row["suite"] = suite
            rows.extend(added)
    rng.shuffle(rows)
    return rows


def build_datasets(exclusions: dict, seed: int = DEFAULT_SEED) -> dict[str, list[dict]]:
    """Author one immutable case order per split, independent of recipe."""
    seen = exclusion_set(exclusions)
    datasets = {}
    seeds = split_seeds(seed)
    for name, groups in SPECIFICATIONS.items():
        rng = random.Random(seeds[name])
        rows = []
        for size, stratum, bad in groups:
            rows.extend(build_rows(name, size, stratum, rng, seen, bad_targets=bad, offset=len(rows)))
        rng.shuffle(rows)
        datasets[name] = rows
    datasets["evaluation"] = build_evaluation(seeds["evaluation"], seen)
    if any(len(datasets[name]) != size for name, size in SPLIT_SIZES.items()):
        raise AssertionError("Wrong authored split size")
```
