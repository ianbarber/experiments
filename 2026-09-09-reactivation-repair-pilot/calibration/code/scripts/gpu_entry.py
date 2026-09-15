"""Prime CUDA before importing model libraries; stages contain no service control."""
import json
from pathlib import Path
import runpy
import sys

if len(sys.argv) < 2:
    raise SystemExit('Usage: gpu_entry.py scripts/model_stage.py [arguments]')
if '--check-only' in sys.argv:
    raise SystemExit('CPU preflight must invoke model_stage.py directly, bypassing gpu_entry.py.')

import torch

config = json.loads((Path(__file__).resolve().parents[1] / 'configs/pilot.json').read_text())
print('GPU launcher: initial free,total', torch.cuda.mem_get_info(), flush=True)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
fraction = config['cuda_allocator_limit_gib'] * 1024**3 / torch.cuda.get_device_properties(0).total_memory
if not 0 < fraction <= 1:
    raise ValueError('Invalid CUDA allocator limit.')
torch.cuda.set_per_process_memory_fraction(fraction)
warmup = torch.zeros(1024, 1024, device='cuda')
torch.cuda.synchronize()
del warmup
print('GPU launcher: primed', torch.cuda.mem_get_info(), flush=True)
script = Path(sys.argv[1])
sys.argv = sys.argv[1:]
sys.path.insert(0, str(script.resolve().parent))
runpy.run_path(str(script), run_name='__main__')
