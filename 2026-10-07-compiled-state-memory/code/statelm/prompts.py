"""Prompt layout shared by all conditions.

compile text  = system turn carrying the memory (fed through the full model once; only GDN state kept)
session text  = user question + non-thinking assistant prefix + answer lead-in (identical across conditions)

Qwen3.8's template injects a "Reasoning effort is set to ..." preamble into the system turn unless
enable_thinking=False, so every render passes it and we assert the preamble is absent.
"""
MEMORY_SYSTEM = "You are a helpful assistant. Here is your memory about the user:\n{memory}"


def _check(s: str) -> str:
    assert "Reasoning effort" not in s and "think carefully" not in s, "thinking preamble leaked into prompt"
    return s


def compile_text(tok, memory: str) -> str:
    # The template refuses a system turn without a user turn, so render system+user and keep the
    # system block (asserted to be an exact prefix of the template's own rendering).
    content = MEMORY_SYSTEM.format(memory=memory)
    sys_block = f"<|im_start|>system\n{content}<|im_end|>\n"
    full = tok.apply_chat_template([{"role": "system", "content": content}, {"role": "user", "content": "x"}],
                                   tokenize=False, add_generation_prompt=False, enable_thinking=False)
    assert full.startswith(sys_block), "chat template system block changed"
    # check only what the template generated (the memory itself may say "think carefully")
    _check(full.replace(content, "", 1))
    return sys_block


def session_text(tok, question: str, lead_in: str = "") -> str:
    s = tok.apply_chat_template([{"role": "user", "content": question}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    _check(s.replace(question, "", 1) + lead_in)
    return s + lead_in


def check_template(tok) -> None:
    """compile_text + session_text must equal the template's own rendering of [system, user]."""
    mem, q = "- [2026-01-01] The user likes tea.", "What do I like?"
    ref = tok.apply_chat_template([{"role": "system", "content": MEMORY_SYSTEM.format(memory=mem)},
                                   {"role": "user", "content": q}], tokenize=False,
                                  add_generation_prompt=True, enable_thinking=False)
    assert compile_text(tok, mem) + session_text(tok, q) == ref, "compile+session != template rendering"
