"""Prime CUDA before the large model-library imports on the shared GB10.

Startup-only workaround; no model weights or experiment RNG settings are changed.
GPU stages use it explicitly, so CPU-only tests do not initialize CUDA.
"""
import runpy,sys,json
from pathlib import Path
import torch
if len(sys.argv)<2:raise SystemExit('Usage: gpu_entry.py scripts/stage.py [arguments]')
print('GPU launcher: initial free,total',torch.cuda.mem_get_info(),flush=True)
config=json.loads((Path(__file__).resolve().parents[1]/'configs/pilot.json').read_text())
torch.manual_seed(0);torch.cuda.manual_seed_all(0)
print('GPU launcher: seeded',torch.cuda.mem_get_info(),flush=True)
torch.cuda.set_per_process_memory_fraction(config['cuda_allocator_limit_gib']*1024**3/torch.cuda.get_device_properties(0).total_memory)
print('GPU launcher: capped',torch.cuda.mem_get_info(),flush=True)
warmup=torch.zeros(1024,1024,device='cuda');torch.cuda.synchronize()
print('GPU launcher: primed',torch.cuda.mem_get_info(),flush=True)
script=Path(sys.argv[1]);sys.argv=sys.argv[1:];sys.path.insert(0,str(script.resolve().parent))
runpy.run_path(str(script),run_name='__main__')
