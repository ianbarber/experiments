"""Public entrypoint defaults to the fixed expansion at data seed 42."""
import sys

from .run import main as run


def configured(arguments):
    arguments = list(arguments)
    supplied = {arg.split("=", 1)[0] for arg in arguments if arg.startswith("--")}
    seed = "42"
    for i, arg in enumerate(arguments):
        if arg == "--seed" and i + 1 < len(arguments):
            seed = arguments[i + 1]
        elif arg.startswith("--seed="):
            seed = arg.split("=", 1)[1]
    defaults = {"--seed": ["42"], "--output": [f"results/expansion-seed{seed}"],
                "--causal-pairs": ["20"], "--query-patch-mode": ["cached"],
                "--candidate-layers": ["19", "23", "24", "26", "27", "28", "30", "31"],
                "--min-answer-accuracy": ["0.9"], "--min-pair-accuracy": ["0.8"]}
    for flag, values in defaults.items():
        if flag not in supplied:
            arguments.extend([flag, *values])
    return arguments


def main():
    sys.argv[1:] = configured(sys.argv[1:])
    run()
