# Code — LLM-JEPA

The scripts as run in the original repo (`ianbarber/llmjepa`, archived). Run them from
this `code/` directory: defaults write hidden-state caches to `data/` (gitignored, ~13 GB)
and checkpoints/results to `experiments/` (the committed result JSONs now live in
`../results/`).

Environment: `pip install -r requirements.txt` (torch, transformers, datasets,
scikit-learn, scipy, matplotlib…). Model `Qwen/Qwen3-4B`. flash-attn was not
installed; sdpa was used.

Pipeline, in order:

1. `src/extract_states.py` — generate GSM8K completions, keep the correct ones, extract
   problem / solution / first-step hidden states at every layer in fp16 (batched).
   ```
   python src/extract_states.py --model Qwen/Qwen3-4B --dataset gsm8k --split train
   ```
2. `src/analysis.py` — per-layer cosine, PCA, ridge probes; optional cross-domain.
   ```
   python src/analysis.py --data_dir data/hidden_states/Qwen_Qwen3-4B_gsm8k_train
   ```
3. `src/mlp_probe.py` — MLP-vs-linear probe diagnostic on the problem→first-step map.
   ```
   python src/mlp_probe.py --data_dir data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train
   ```
4. `src/train_jepa.py` — Phase 1 predictor plus mean / identity / ridge / k-NN baselines.
   ```
   python src/train_jepa.py --data_dir data/hidden_states_v2/Qwen_Qwen3-4B_gsm8k_train --layer 21 --loss mse
   ```
5. `src/inject_jepa.py` — Phase 2 alpha sweep, K sweep, and the controlled evaluation.
   ```
   python src/inject_jepa.py --experiment p2.1
   python src/inject_jepa.py --experiment p2.2 --alphas 0.001
   python src/inject_jepa.py --experiment p2.3 --alphas 0.001 --Ks 5
   ```

Each script's module docstring lists the remaining flags.
