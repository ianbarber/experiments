"""Tokenization, target-only objective, and explicitly configured local model I/O.

Importing this module does not import PyTorch or initialize CUDA. CPU preflight
loads only the local tokenizer; the stage launcher owns GPU initialization.
"""
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/pilot.json').read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def sha_row(row):
    return hashlib.sha256(json.dumps(row, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False).encode()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def schedule_seed(seed, purpose, identity):
    """Stable independent streams; never use Python's process-randomized hash."""
    value = sha_row({'seed': seed, 'purpose': purpose, 'identity': identity})
    return int(value[:16], 16) % (2**31 - 1)


def row_sampling_seed(row, seed):
    value = row.get('sampling_seed')
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            raise ValueError('sampling_seed must be an integer in [0, 2**32).')
        return value
    return schedule_seed(seed, 'generation', {'id': row['id'], 'draw_index': row.get('draw_index')})


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve(CONFIG['model_path']), local_files_only=True)
    tok.padding_side = 'left'
    if tok.eos_token_id is None:
        raise ValueError('The frozen tokenizer must have an EOS token.')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def messages(row):
    from task import SYSTEM
    result = row.get('messages') or [
        {'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': row['prompt']}]
    if not isinstance(result, list) or not result or result[-1]['role'] != 'user':
        raise ValueError('Model prefixes must end with a user message.')
    return result


def prefix_ids(tok, msgs):
    result = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    if isinstance(result, dict) or hasattr(result, 'input_ids'):
        result = result['input_ids']
    if not isinstance(result, list) or not result or not all(isinstance(x, int) for x in result):
        raise ValueError('Expected a nonempty flat token list from the chat template.')
    return result


def encode_example(tok, msgs, target):
    if not isinstance(target, str) or not target.strip():
        raise ValueError('Training targets must be nonempty text.')
    prefix = prefix_ids(tok, msgs)
    target_ids = tok.encode(target, add_special_tokens=False)
    # Prevent embedded control tokens from silently changing target boundaries.
    if any(token in set(tok.all_special_ids) for token in target_ids):
        raise ValueError('A target contains an embedded tokenizer control token.')
    target_ids += [tok.eos_token_id]
    # Labels are already shifted: the final prefix position predicts target[0].
    ids = prefix + target_ids[:-1]
    labels = [-100] * (len(prefix) - 1) + target_ids
    if len(ids) > CONFIG['max_sequence_length']:
        raise ValueError(f'No silent truncation: training example has {len(ids)} tokens.')
    if len(ids) != len(labels) or sum(x != -100 for x in labels) != len(target_ids):
        raise ValueError('Target alignment/masking invariant failed.')
    return {'input_ids': ids, 'labels': labels, 'prefix_tokens': len(prefix),
            'target_tokens': len(target_ids)}


def collate(examples, pad, device='cuda'):
    import torch
    width = max(len(example['input_ids']) for example in examples)
    ids, labels, masks = [], [], []
    for example in examples:
        padding = width - len(example['input_ids'])
        ids.append([pad] * padding + example['input_ids'])
        labels.append([-100] * padding + example['labels'])
        masks.append([0] * padding + [1] * len(example['input_ids']))
    mask = torch.tensor(masks, device=device)
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    return {'input_ids': torch.tensor(ids, device=device), 'attention_mask': mask,
            'position_ids': positions}, torch.tensor(labels, device=device)


def token_losses(model, inputs, labels):
    """No detach or prefix loss. Gradients can pass through prefix activations."""
    import torch
    active = labels.ne(-100)
    first = active.any(0).nonzero()[0].item()
    suffix = labels.shape[1] - first
    logits = model(**inputs, use_cache=False, logits_to_keep=suffix).logits.float()
    target = labels[:, -suffix:]
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), target.reshape(-1),
        reduction='none', ignore_index=-100).reshape(target.shape)
    return loss, target.ne(-100)


def generation_settings(tok, sample, max_new_tokens):
    """Construct a fresh config instead of inheriting model generation defaults."""
    return dict(do_sample=bool(sample), temperature=0.7 if sample else 1.0,
                top_p=0.95 if sample else 1.0, top_k=0, repetition_penalty=1.0,
                max_new_tokens=max_new_tokens, num_beams=1, num_return_sequences=1,
                min_new_tokens=0, pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, bos_token_id=tok.bos_token_id,
                use_cache=True, return_dict_in_generate=False)


def load_model(adapter=None, train=False, seed=42):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    seed_all(seed)
    torch.set_num_threads(8)
    fraction = CONFIG['cuda_allocator_limit_gib'] * 1024**3 / torch.cuda.get_device_properties(0).total_memory
    if not 0 < fraction <= 1:
        raise ValueError('Invalid CUDA allocator limit for this GPU.')
    torch.cuda.set_per_process_memory_fraction(fraction)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        resolve(CONFIG['model_path']), dtype=torch.bfloat16, quantization_config=quant,
        device_map={'': 'cuda'}, attn_implementation='sdpa', local_files_only=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train,
                                          gradient_checkpointing_kwargs={'use_reentrant': False})
    if adapter:
        model = PeftModel.from_pretrained(model, str(resolve(adapter)), is_trainable=train)
    elif train:
        model = get_peft_model(model, LoraConfig(
            r=CONFIG['lora_rank'], lora_alpha=CONFIG['lora_alpha'],
            lora_dropout=CONFIG['lora_dropout'], target_modules=CONFIG['lora_targets'],
            task_type='CAUSAL_LM'))
    if train:
        model.train()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
    else:
        model.eval()
    return model


def generate_batch(model, tok, list_messages, max_new_tokens=192, sample=False):
    import torch
    from transformers import GenerationConfig
    sequences = [prefix_ids(tok, item) for item in list_messages]
    width = max(map(len, sequences))
    if width + max_new_tokens > CONFIG['max_sequence_length']:
        raise ValueError('Generation prefix plus reserved output exceeds the sequence limit.')
    ids = torch.tensor([[tok.pad_token_id] * (width - len(x)) + x for x in sequences], device='cuda')
    mask = torch.tensor([[0] * (width - len(x)) + [1] * len(x) for x in sequences], device='cuda')
    # GenerationMixin computes and extends position IDs from this left-padding
    # mask. Passing a fixed position_ids tensor can fail when the cache grows.
    settings = generation_settings(tok, sample, max_new_tokens)
    with torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=mask,
                             generation_config=GenerationConfig(**settings))
    generated = out[:, width:].cpu().tolist()
    details = []
    for sequence in generated:
        ended = tok.eos_token_id in sequence
        if ended:
            sequence = sequence[:sequence.index(tok.eos_token_id) + 1]
        before_eos = sequence[:-1] if ended else sequence
        unexpected = [token for token in before_eos if token in set(tok.all_special_ids)]
        details.append({'text': tok.decode(sequence, skip_special_tokens=True),
                        'text_with_special_tokens': tok.decode(sequence, skip_special_tokens=False),
                        'token_ids': sequence, 'generated_tokens': len(sequence),
                        'unexpected_special_token_ids': unexpected,
                        'finish_reason': 'eos' if ended else 'length'})
    return details
