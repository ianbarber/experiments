from fwlayer.cli import configured


def test_public_cli_defaults_to_the_fixed_seed42_expansion():
    args=configured([])
    assert args[args.index("--seed")+1]=="42"
    assert args[args.index("--causal-pairs")+1]=="20"
    assert args[args.index("--query-patch-mode")+1]=="cached"


def test_explicit_seed_and_mode_override_the_preset():
    args=configured(["--seed","314159","--query-patch-mode","full"])
    assert args.count("--seed")==1
    assert args[args.index("--query-patch-mode")+1]=="full"
    assert args[args.index("--output")+1]=="results/expansion-seed314159"
