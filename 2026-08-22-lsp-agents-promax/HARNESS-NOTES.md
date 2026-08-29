# Two harness findings that transfer beyond this experiment

Extracted from the LSP study (see REPORT.md) because they apply to anyone running
mini-swe-agent-style evaluations, especially against slow/local model servers.

## 1. mini-swe-agent's 2-hour container wall masquerades as step-cap exhaustion

`environment.container_timeout` defaults to `"2h"` — instance containers are started
with `sleep 2h`. An episode that outlives it loses its container mid-flight: every
subsequent command returns "Error response from daemon: No such container", and the
agent grinds its remaining step budget against a dead environment (we logged one episode
making 188 submission attempts against a corpse). The episode then reports
`LimitsExceeded`, indistinguishable in exit status from genuine step-cap exhaustion.

Why it bites unevenly: episode wall-clock scales inversely with decode speed. At our
local ~15 tok/s per stream, 2 hours ≈ 60–90 agent steps of budget; at hosted API speeds
the same 300-step episode rarely approaches it. Any *condition* that lengthens episodes
(in our case: acceptance-gate repair loops) eats disproportionately more wall deaths —
container-death counts across our five local arms were monotone in mean episode length
(4 → 16 per 58 episodes), which confounded the arm comparison until fixed.

Diagnosis: grep trajectories for "No such container". Fix: set
`environment.container_timeout` ≥ your worst-case episode wall-clock (we used `8h`;
deaths went to zero).

## 2. The serving stack is a first-order variable: 72% → 94% on identical everything else

Same model (Qwen3.8-27B), same weights precision class (fp8), same scaffold, prompts,
sampling parameters, and instances: our tuned local stack (SGLang container for GB10,
DSpark speculative decoding, fp8 KV cache, sm121 triton kernels) resolved 72% [58–83];
OpenRouter fp8 endpoints (CoreWeave/Parasail) resolved 94% [84–98]. The container-wall
fix accounts for ~2 of the ~11 instances of gap.

We could not fully separate the two remaining causes — local serving quality loss vs
reasoning-effort defaults (`chat_template_kwargs`/`reasoning.effort` do **not** pass
through OpenRouter, so hosted ran the model's default xhigh thinking while local pinned
medium) — and the hosted number also carries a contamination signal (REPORT.md §7). But
the operational lesson stands regardless of attribution: **before crediting or blaming an
intervention measured on a local serving stack, calibrate that stack against a reference
endpoint on the same instances.** A 20-point serving delta is larger than any
intervention effect we measured in seven arm-configurations.

Corollary for cost planning: reasoning-effort passthrough failing "harmlessly" is not
harmless — hosted reasoning bills as output tokens. Probe 1–2 episodes and read actual
billed usage before launching a fleet (we missed this and burned ~15–20× the estimate).
