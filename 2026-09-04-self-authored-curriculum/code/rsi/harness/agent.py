"""RsiAgent: the single-bash-action solver loop, run by Harbor inside a task container (docs/03 §5).

Harbor gives us the environment (build, no-network sandbox, user, verifier); we own the policy calls,
the shell semantics and the per-turn records that the trainer consumes verbatim
(vLLM's prompt_token_ids + choices[0].token_ids + logprobs; the trainer never re-renders a prompt).
"""
from __future__ import annotations
import asyncio, json, os, time
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from rsi.config import HarnessDefaults
from rsi.harness.prompt import (STATE_DIR, COMPACT_REQUEST, OBS_TEMPLATE, RESUME_TEMPLATE, build_exec_command,
                                parse_action, render_system_prompt, split_exec_output, split_thinking, strip_thinking, truncate_observation)
from rsi.harness.vllm_client import VLLMChat

def _b(v, default: bool) -> bool:
    if v is None: return default
    if isinstance(v, bool): return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")

_EXECUTOR = None
def _ensure_executor():
    """asyncio.to_thread uses the loop's default executor (min(32, cpu+4) = 20 workers here); 32+ concurrent trials need more."""
    global _EXECUTOR
    if _EXECUTOR is None:
        from concurrent.futures import ThreadPoolExecutor
        _EXECUTOR = ThreadPoolExecutor(max_workers=96, thread_name_prefix="rsi-llm")
        asyncio.get_running_loop().set_default_executor(_EXECUTOR)

class RsiAgent(BaseAgent):
    SUPPORTS_ATIF = False

    @staticmethod
    @override
    def name() -> str:
        return "rsi-agent"

    @override
    def version(self) -> str:
        return "0.1.0"

    def __init__(self, logs_dir: Path, model_name: str | None = None, *, base_url: str = "http://127.0.0.1:8000/v1",
                 model: str = "trainee", max_turns=None, max_tokens=None, window=None, summary_cap=None, retained_turns=None,
                 max_compactions=None, command_timeout=None, obs_cap=None, temperature=None, top_p=None,
                 thinking=None, brevity=None, seed=None, policy_tag: str = "", logprobs=None, **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        d = HarnessDefaults()
        self.cfg = HarnessDefaults(
            max_turns=int(max_turns or d.max_turns), max_tokens_per_turn=int(max_tokens or d.max_tokens_per_turn),
            window_tokens=int(window or d.window_tokens), summary_cap_tokens=int(summary_cap or d.summary_cap_tokens),
            retained_turns=int(retained_turns or d.retained_turns), max_compactions=int(max_compactions or d.max_compactions),
            command_timeout_sec=int(command_timeout or d.command_timeout_sec), observation_cap_chars=int(obs_cap or d.observation_cap_chars),
            temperature=float(temperature if temperature is not None else d.temperature), top_p=float(top_p if top_p is not None else d.top_p),
            enable_thinking=_b(thinking, d.enable_thinking), brevity_sentence=_b(brevity, d.brevity_sentence))
        self.seed = int(seed) if seed not in (None, "", "none") else None
        self.policy_tag = policy_tag
        self.logprobs = _b(logprobs, True)
        self.llm = VLLMChat(base_url, model)
        self.tools_text = ""
        self._records_path = Path(logs_dir) / "turns.jsonl"
        self._summary_path = Path(logs_dir) / "episode.json"

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        # under heavy load `docker compose exec` itself can take tens of seconds to start; be patient and retry once
        last = None
        for attempt in range(3):
            try:
                r = await environment.exec(f"mkdir -p {STATE_DIR} && cat /etc/rsi-tools.txt 2>/dev/null || true", timeout_sec=180)
                self.tools_text = (r.stdout or "").strip() or "(standard Ubuntu 24.04 tools; python3, node, gcc, sqlite3, jq, git)"
                return
            except Exception as ex:
                last = ex; await __import__("asyncio").sleep(5 * (attempt + 1))
        raise RuntimeError(f"agent setup failed after retries: {last}")

    # ------------------------------------------------------------------ helpers
    def _record(self, rec: dict):
        self._records_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._records_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    async def _run_command(self, environment: BaseEnvironment, cmd: str) -> tuple[str, int | None, str | None, float]:
        t0 = time.time()
        wrapped = build_exec_command(cmd, self.cfg.command_timeout_sec)
        try:
            r = await environment.exec(wrapped, timeout_sec=self.cfg.command_timeout_sec + 30)
            raw = (r.stdout or "") + (("\n" + r.stderr) if r.stderr else "")
        except Exception as ex:  # harness-level failure (docker exec died, timeout at the compose layer)
            raw = f"[harness] command execution failed: {type(ex).__name__}: {ex}"
        out, rc, cwd = split_exec_output(raw)
        return out, rc, cwd, time.time() - t0

    def _observation(self, out: str, rc: int | None, cwd: str | None, latency: float) -> str:
        body = truncate_observation(out, self.cfg.observation_cap_chars) if out else "(no output)"
        extra = ""
        if cwd: extra += f" cwd={cwd}"
        if rc == 124: extra += f" timed_out_after={self.cfg.command_timeout_sec}s"
        return OBS_TEMPLATE.format(exit_code=rc if rc is not None else "?", extra=extra, body=body)

    # ------------------------------------------------------------------ main loop
    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        _ensure_executor()
        cfg = self.cfg
        t_start = time.time()
        system = render_system_prompt(self.tools_text, cfg.command_timeout_sec, cfg.max_turns, cfg.brevity_sentence)
        first_user = f"Task:\n\n{instruction.strip()}\n\nYou are in /app. Begin."
        messages = [{"role": "system", "content": system}, {"role": "user", "content": first_user}]
        segment = 0; turn = 0; total_turns = 0; compactions = 0
        n_in = n_out = 0; no_cmd_streak = 0; stop_reason = "max_turns"
        last_prompt_len = 0; last_gen = 0; last_obs_chars = 0
        transcript = []  # (assistant_content, observation) pairs of the current segment, for retention after compaction

        def est_next_prompt() -> int:
            return last_prompt_len + last_gen + last_obs_chars // 3 + 16

        while total_turns < cfg.max_turns:
            # ---- prospective compaction (docs/03 §5)
            if turn > 0 and est_next_prompt() + cfg.max_tokens_per_turn + cfg.summary_cap_tokens > cfg.window_tokens:
                if compactions >= cfg.max_compactions:
                    stop_reason = "max_compactions"; break
                req = messages + [{"role": "user", "content": COMPACT_REQUEST.format(cap=cfg.summary_cap_tokens)}]
                t0 = time.time()
                res = await asyncio.to_thread(self.llm.complete, req, max_tokens=cfg.summary_cap_tokens + 512, temperature=cfg.temperature,
                                              top_p=cfg.top_p, enable_thinking=cfg.enable_thinking, seed=self.seed, logprobs=self.logprobs)
                n_in += res.prompt_tokens; n_out += res.completion_tokens
                summary = strip_thinking(res.content) or "(no summary produced)"
                self._record({"segment": segment, "turn": turn, "is_summary": True, "prompt_token_ids": res.prompt_token_ids,
                              "token_ids": res.token_ids, "logprobs": res.logprobs, "finish_reason": res.finish_reason,
                              "content": res.content, "latency_s": round(time.time() - t0, 3), "ts": time.time()})
                compactions += 1; segment += 1; turn = 0
                keep = transcript[-cfg.retained_turns:]
                messages = [{"role": "system", "content": system},
                            {"role": "user", "content": first_user + "\n\n" + RESUME_TEMPLATE.format(summary=summary)}]
                for a, o in keep:
                    messages += [{"role": "assistant", "content": a}, {"role": "user", "content": o}]
                transcript = list(keep)
                last_prompt_len = 0; last_gen = 0; last_obs_chars = 0
            # ---- policy step
            t0 = time.time()
            # off the event loop: a blocking HTTP call here would serialise every trial in the Harbor process
            res = await asyncio.to_thread(self.llm.complete, messages, max_tokens=cfg.max_tokens_per_turn, temperature=cfg.temperature,
                                          top_p=cfg.top_p, enable_thinking=cfg.enable_thinking, seed=self.seed, logprobs=self.logprobs)
            latency_llm = time.time() - t0
            n_in += res.prompt_tokens; n_out += res.completion_tokens
            last_prompt_len = len(res.prompt_token_ids) or res.prompt_tokens; last_gen = len(res.token_ids) or res.completion_tokens
            if turn == 0 and segment == 0 and last_prompt_len + cfg.max_tokens_per_turn + cfg.summary_cap_tokens > cfg.window_tokens:
                stop_reason = "prompt_too_long_for_window"
                self._record({"segment": 0, "turn": 0, "error": stop_reason, "prompt_tokens": last_prompt_len, "ts": time.time()})
                break
            cmd, done = parse_action(res.content)
            rec = {"segment": segment, "turn": turn, "is_summary": False, "prompt_token_ids": res.prompt_token_ids,
                   "token_ids": res.token_ids, "logprobs": res.logprobs, "finish_reason": res.finish_reason,
                   "content": res.content, "command": cmd, "llm_latency_s": round(latency_llm, 3), "ts": time.time()}
            reasoning, visible = split_thinking(res.content)
            assistant_content = visible if visible.strip() else "(empty)"   # history holds the visible reply only (the template drops old reasoning anyway)
            rec["reasoning_chars"] = len(reasoning)
            if done and cmd is None:
                rec.update({"done": True}); self._record(rec)
                messages.append({"role": "assistant", "content": assistant_content})
                stop_reason = "task_complete"; total_turns += 1; break
            if cmd is None:
                no_cmd_streak += 1
                obs = ("<observation exit=? note=\"no command found\">\nReply with your reasoning and exactly one ```bash block, "
                       "or the single line TASK_COMPLETE when finished.\n</observation>")
                rec.update({"exit_code": None, "observation": obs}); self._record(rec)
                messages += [{"role": "assistant", "content": assistant_content}, {"role": "user", "content": obs}]
                transcript.append((assistant_content, obs))
                last_obs_chars = len(obs); turn += 1; total_turns += 1
                if no_cmd_streak >= 3 or res.finish_reason == "length" and no_cmd_streak >= 2:
                    stop_reason = "no_command"; break
                continue
            no_cmd_streak = 0
            out, rc, cwd, latency_cmd = await self._run_command(environment, cmd)
            obs = self._observation(out, rc, cwd, latency_cmd)
            rec.update({"exit_code": rc, "cwd": cwd, "observation": obs, "cmd_latency_s": round(latency_cmd, 3), "obs_chars": len(obs)})
            self._record(rec)
            messages += [{"role": "assistant", "content": assistant_content}, {"role": "user", "content": obs}]
            transcript.append((assistant_content, obs))
            last_obs_chars = len(obs); turn += 1; total_turns += 1

        summary = {"turns": total_turns, "segments": segment + 1, "compactions": compactions, "stop_reason": stop_reason,
                   "n_input_tokens": n_in, "n_output_tokens": n_out, "wall_s": round(time.time() - t_start, 2),
                   "config": self.cfg.__dict__, "seed": self.seed, "policy_tag": self.policy_tag, "model": self.llm.model}
        self._summary_path.parent.mkdir(parents=True, exist_ok=True)
        self._summary_path.write_text(json.dumps(summary, indent=1))
        context.n_input_tokens = n_in; context.n_output_tokens = n_out
        context.metadata = summary
