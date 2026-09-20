# Code

Extraction, vector fitting, steering ladders, the two-button engine and the analysis. Paths
below are relative to this directory.

## Layout

| Path | What it is |
|---|---|
| `itchlib/common.py` | Model table, paths from the environment, the bridge that imports the paper's scripts as modules, the paper's random-direction recipe |
| `itchlib/extract.py` | Final-token residual activations at every decoder block output, via forward hooks |
| `itchlib/vectors.py` | Concept vectors fitted with the paper's own `compute_pain_vector`, AUC and 5-fold CV functions; their control-vector recipe |
| `itchlib/lexicon.py` | Keyword measures, committed before any steered text existed. The paper's pain/hurt regex is kept verbatim |
| `itchlib/buttons.py` | The two-button engine: a port of the paper's `04_selfmed_two_buttons.py`. Carried-over functions are marked `[paper]`, changes `[ours]` |
| `data/build_itch_dataset.py`, `data/itch_dataset.json` | The itch dataset: 100 itch sentences, 100 matched "present, not itching" sentences, two written control categories, three reused from the paper (MIT) |
| `data/itch_vivid_prompts.json` | 20 vivid itch prompts for the natural-range check |
| `scripts/00`-`07` | BOS check, extraction, vectors, ladder, ladder report, natural range, buttons, dose probe |
| `scripts/08`-`11` | Button tables and transcripts, the locked hypothesis tests, results assembly, compact tables |
| `scripts/12_report_numbers.py` | Re-derives every number quoted in the README and report from `../results/` only; writes `report_numbers.json/.md`, derived tables and `../images/first_choice_distribution.png` |
| `run_full.sh` | Every button stage in priority order, resumable |
| `doses.json` | The locked doses and itch vector |
| `docs/results_prose.md` | The hand-written half of the generated results file |

## Environment

One GPU with room for Qwen2.5-32B in bf16 plus a KV cache (the run used a GB10 with 128 GB of
unified memory; on that machine nothing else can be resident while the 32B loads). Python 3.12.

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install transformers==5.17.0 peft==0.21.0 accelerate scikit-learn scipy pandas matplotlib safetensors tabulate

git clone https://github.com/valen-research/Pain-axis && git -C Pain-axis checkout 8d1649c
# adapters: extract adapter_Qwen_2.5_{7B,32B}_instruct.tar.gz from hf.co/Valen92/pain-adapters
# into adapters/Qwen_2.5_7B_instruct/ and adapters/Qwen_2.5_32B_instruct/

export ITCH_32B_PATH=/path/to/Qwen2.5-32B-Instruct   # optional local snapshots; default is the HF hub id
export ITCH_7B_PATH=/path/to/Qwen2.5-7B-Instruct
export ITCH_RESULTS=$PWD/results                     # raw outputs, about 2.5 GB with activations
```

## Running it

```bash
M=32b   # or 7b
python data/build_itch_dataset.py
python scripts/00_bos_check.py --model 7b
python scripts/01_extract_activations.py --model $M --tag pain Pain-axis/datasets/3.1_pain_and_control_datasets.json Pain-axis/datasets/3.1_sadness_dataset.json
python scripts/01_extract_activations.py --model $M --tag itch data/itch_dataset.json
python scripts/02_build_vectors.py --model $M
python scripts/03_steering_ladder.py --model $M --vectors pain itch_A_at_pain_layer itch_A rand4817 sadness
python scripts/04_ladder_report.py --model $M
python scripts/05_natural_range.py --model $M --adapter --vectors pain itch_A_at_pain_layer
python scripts/07_dose_probe.py --model $M --vectors pain itch_A_at_pain_layer rand4817 rand2903
./run_full.sh                                   # about 14 h on the GB10; rerun after any interruption
python scripts/08_button_analysis.py --model $M --tag full
python scripts/09_hypotheses.py --model $M
python scripts/10_make_results.py
python scripts/11_compact_tables.py --out ../results
python scripts/12_report_numbers.py              # needs only ../results; no raw JSONL, no GPU
```

Scripts 02, 04 and 08-11 read only saved activations or raw JSONL. GPU scripts append JSONL as
they go and skip completed rows on restart. Seeds are fixed. `06_buttons.py --no-kv-reuse
--no-fork` runs trials the paper's way (every turn re-encoded, arms run separately); the
equivalence check is in the lab notes.

The raw button logs (about 250 MB) are not in this repository. `../results/` has the generated
tables, the machine-readable hypothesis tests, and two compact tables
(`first_choice_per_scenario.csv.gz`, `trials_compact.csv.gz`) from which the first-choice and
press-again statistics can be recomputed independently.
