"""The L0 solver contract: fixed system prompt, observation format, command parsing (docs/03 §5).
Nothing here is model-editable; it is identical at every rollout and every evaluation."""
from __future__ import annotations
import re

SYSTEM_PROMPT = """You are an autonomous engineer working in a Linux shell to complete a task. You cannot ask questions; nobody will answer.

How to work:
- Each reply must contain your brief reasoning and then EXACTLY ONE bash command in a fenced block:
```bash
<command>
```
- The command runs in /app (or the last directory you cd'd into) with a {timeout}s timeout. Environment variables you export and directories you cd into persist between turns. Start long-running servers with `nohup ... &`.
- You see the command's combined output (truncated if long) and its exit code as an <observation>.
- Inspect before you act: read the relevant files, run things, check results. Write files with heredocs (cat > file <<'EOF' ... EOF) or python.
- There is no internet. The tools below are already installed. If you truly need another package, run `pkg install <name>`; it is served from a local mirror when available.
- When the task is fully done and verified as best you can, reply with the single line TASK_COMPLETE and no command. You have at most {max_turns} commands.{brevity}

Installed tools:
{tools}
"""

BREVITY = "\n- Keep your thinking brief: decide, then act."

OBS_TEMPLATE = "<observation exit={exit_code}{extra}>\n{body}\n</observation>"
RESUME_TEMPLATE = ("Your context was reset to save space. You are continuing the same task. Here is the summary you wrote of your progress so far:\n\n"
                   "{summary}\n\nThe shell state (cwd, files, exported variables) is unchanged. Continue from here.")
COMPACT_REQUEST = ("STOP. Before continuing, write a structured summary of your work so far so you can resume after a context reset. "
                   "Use exactly these headings: Goal / Progress / Key decisions / Relevant files and commands / Next steps. "
                   "Be concrete (paths, values, what is verified). At most {cap} tokens. Do not include a bash command.")

_FENCE = re.compile(r"```(?:bash|sh|shell)?[ \t]*\n(.*?)```", re.DOTALL)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)

def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, visible). Qwen3.5 puts the opening <think> in the prompt, so generated text usually holds
    only the closing tag: everything up to the LAST </think> is reasoning."""
    text = text or ""
    if "</think>" in text:
        head, _, tail = text.rpartition("</think>")
        return head.replace("<think>", "", 1).strip(), tail.strip()
    return "", _THINK.sub("", text).strip()

def strip_thinking(text: str) -> str:
    return split_thinking(text)[1]

def parse_action(content: str) -> tuple[str | None, bool]:
    """Return (command, done). done=True when the reply says TASK_COMPLETE with no command.
    Uses the LAST fenced bash block so a model that quotes an earlier command in prose is not misread."""
    visible = strip_thinking(content)
    blocks = _FENCE.findall(visible)
    if blocks:
        cmd = blocks[-1].strip("\n")
        return (cmd if cmd.strip() else None), False
    done = bool(re.search(r"^\s*TASK_COMPLETE\s*$", visible, re.MULTILINE))
    return None, done

def truncate_observation(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    head = int(cap * 0.6); tail = cap - head
    return text[:head] + f"\n... [{len(text) - cap} chars truncated] ...\n" + text[-tail:]

def render_system_prompt(tools: str, timeout: int, max_turns: int, brevity: bool) -> str:
    return SYSTEM_PROMPT.format(tools=tools.strip(), timeout=timeout, max_turns=max_turns, brevity=BREVITY if brevity else "")

# Wrapper that gives the model a persistent-feeling shell through Harbor's stateless exec:
# cwd and exported variables are saved after every command and restored before the next.
STATE_DIR = "/tmp/.rsi"
EXEC_WRAPPER = r"""mkdir -p {sd}; cat > {sd}/cmd <<'{eof}'
{cmd}
{eof}
timeout -k 5 {timeout} bash -c '
cd "$(cat {sd}/cwd 2>/dev/null || echo /app)" 2>/dev/null || cd /app
[ -f {sd}/env ] && . {sd}/env 2>/dev/null
. {sd}/cmd
rc=$?
pwd > {sd}/cwd
export -p | grep -vE "^declare -x (PWD|OLDPWD|SHLVL|_|RSI_[A-Z_]*)=" > {sd}/env 2>/dev/null
exit $rc' 2>&1
rc=$?
echo "__RSI_RC__=$rc __RSI_CWD__=$(cat {sd}/cwd 2>/dev/null || echo /app)"
"""

def build_exec_command(cmd: str, timeout: int) -> str:
    eof = "RSI_CMD_EOF_7f3a"
    while eof in cmd:
        eof += "x"
    return EXEC_WRAPPER.format(sd=STATE_DIR, eof=eof, cmd=cmd, timeout=timeout)

_TRAILER = re.compile(r"__RSI_RC__=(-?\d+) __RSI_CWD__=(\S*)\s*$")

def split_exec_output(raw: str) -> tuple[str, int | None, str | None]:
    """Separate the command output from the wrapper trailer. Returns (output, exit_code, cwd)."""
    raw = raw or ""
    m = _TRAILER.search(raw)
    if not m:
        return raw.rstrip(), None, None
    return raw[: m.start()].rstrip(), int(m.group(1)), m.group(2)
