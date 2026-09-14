"""Start / stop / wait for the vLLM rollout server (serve venv), always on the fixed merged path or the base."""
from __future__ import annotations
import os, signal, subprocess, time
from pathlib import Path
import httpx
from rsi import config as C

def start(model_path: Path | str, *, port: int = C.VLLM_PORT, max_num_seqs: int = 32, kv_dtype: str | None = None,
          log_path: Path | None = None, gpu_util: float = 0.90, dtype: str = "bfloat16",
          prefix_caching: bool = True) -> subprocess.Popen:
    env = dict(os.environ, VLLM_CACHE_ROOT=os.environ.get("VLLM_CACHE_ROOT", "${HOME}/.cache/vllm-rsi"))
    cmd = [str(C.VENVS["serve"] / "bin" / "vllm"), "serve", str(model_path), "--served-model-name", C.SERVED_MODEL_NAME,
           "--port", str(port), "--dtype", dtype, "--language-model-only",
           "--enable-prefix-caching" if prefix_caching else "--no-enable-prefix-caching",
           "--max-model-len", "32768", "--max-num-seqs", str(max_num_seqs), "--gpu-memory-utilization", str(gpu_util),
           "--hf-overrides", '{"head_dtype": "float32"}', "--logprobs-mode", "processed_logprobs",
           "--return-tokens-as-token-ids", "--generation-config", "vllm", "--no-enable-log-requests"]
    if kv_dtype: cmd += ["--kv-cache-dtype", kv_dtype]
    lf = open(log_path or Path("/tmp/vllm-rsi.log"), "w")
    return subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, start_new_session=True)

def wait_healthy(port: int = C.VLLM_PORT, timeout_s: float = 1800) -> float:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            if httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200:
                return time.time() - t0
        except httpx.HTTPError:
            pass
        time.sleep(3)
    raise TimeoutError(f"vLLM on :{port} not healthy after {timeout_s}s")

def is_up(port: int = C.VLLM_PORT) -> bool:
    try: return httpx.get(f"http://127.0.0.1:{port}/health", timeout=3).status_code == 200
    except httpx.HTTPError: return False

def stop(proc: subprocess.Popen | None = None, *, wait_s: float = 30) -> None:
    """Stop the server (by handle, or any `vllm serve` on the box) and wait for the GPU memory to be released."""
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
    subprocess.run(["pkill", "-f", "vllm serv[e]"], check=False)
    t0 = time.time()
    while time.time() - t0 < wait_s and subprocess.run(["pgrep", "-f", "vllm serv[e]"], capture_output=True).returncode == 0:
        time.sleep(2)
    subprocess.run(["pkill", "-9", "-f", "vllm serv[e]"], check=False)
    time.sleep(3)

def gpu_mem_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0]) if out.strip() else -1
