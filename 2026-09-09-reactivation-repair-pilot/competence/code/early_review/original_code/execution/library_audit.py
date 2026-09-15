"""Read exact installed package sources using stdlib; no package/model imports."""
import ast
from datetime import datetime, timezone
import hashlib
from importlib.metadata import distribution, version
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SOURCES = {
 'transformers/generation/configuration_utils.py': [(368, 469), (517, 540), (571, 616)],
 'transformers/generation/utils.py': [(713, 735), (1106, 1126), (1235, 1271), (1295, 1298),
                                      (1661, 1675), (1711, 1754), (2478, 2484), (2804, 2857)],
 'transformers/generation/logits_process.py': [(306, 326), (357, 375), (406, 413)],
 'transformers/models/qwen2/modeling_qwen2.py': [(430, 485)],
 'peft/peft_model.py': [(464, 485), (522, 525), (540, 580), (920, 926), (1353, 1374), (2035, 2056)],
 'peft/utils/other.py': [(130, 179)],
 'peft/tuners/lora/layer.py': [(246, 266)],
 'torch/optim/adamw.py': [(21, 66)],
}
out = {'created_utc': datetime.now(timezone.utc).isoformat(),
       'method': 'stdlib distribution metadata and source reads only; no torch/transformers/PEFT import',
       'versions': {name: version(name) for name in ('torch', 'transformers', 'peft', 'bitsandbytes')},
       'sources': {}}
for rel, ranges in SOURCES.items():
    pkg = rel.split('/')[0]
    p = Path(distribution(pkg).locate_file(rel))
    data = p.read_bytes(); lines = data.decode().splitlines()
    out['sources'][rel] = {'sha256': hashlib.sha256(data).hexdigest(),
                          'excerpts': [{'first_line': start, 'last_line': end,
                                        'text': '\n'.join(f'{i+1}: {lines[i]}' for i in range(start-1, end))}
                                       for start, end in ranges]}
    if rel.endswith('configuration_utils.py'):
        tree = ast.parse(data)
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                      and n.name == '_get_default_generation_params')
        ret = next(n for n in method.body if isinstance(n, ast.Return))
        out['library_generation_fallbacks'] = ast.literal_eval(ret.value)
        klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'GenerationConfig')
        init = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
        defaults = {}
        for n in init.body:
            if (isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'pop'
                    and len(n.value.args) == 2 and isinstance(n.targets[0], ast.Attribute)):
                defaults[n.targets[0].attr] = ast.literal_eval(n.value.args[1])
        out['constructor_generation_defaults'] = defaults
base_path = Path('/models/Qwen2.5-3B-Instruct/generation_config.json')
out['base_generation_config'] = json.loads(base_path.read_text())
out['base_generation_config_exact_text'] = base_path.read_text()
out['base_generation_config_sha256'] = hashlib.sha256(base_path.read_bytes()).hexdigest()
for sample in (False, True):
    values = dict(out['constructor_generation_defaults'], **out['library_generation_fallbacks'])
    values.update(out['base_generation_config'])
    values.update(do_sample=sample, max_new_tokens=192, pad_token_id=151643, eos_token_id=151645, use_cache=True)
    if sample:
        values.update(temperature=0.7, top_p=0.95)
    out['effective_sampling' if sample else 'effective_greedy'] = values
out['effective_dictionary_scope'] = 'Resolved configuration before max_length preparation; max_length=20 is replaced by padded_input_length+max_new_tokens=192. Sampling warpers are inactive when do_sample=False. None-valued optional constraints are absent.'
print(json.dumps(out, indent=2))
