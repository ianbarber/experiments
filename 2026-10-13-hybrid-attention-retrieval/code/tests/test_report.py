from fwlayer.report import causal_stats


def row(distance, index, fraction):
    return {"pair_id": f"d{distance}-p{index:03d}", "fraction_of_counterfactual_gap": fraction,
            "correct_margin_change": -fraction}


def test_repeated_distances_are_one_example_cluster():
    stats = causal_stats([row(d, 0, .5) for d in (128, 1024, 4096, 8192)])
    assert stats["n_pairs"] == 4
    assert stats["n_example_clusters"] == 1
    assert stats["bootstrap_95_interval"] is None


def test_patch_directions_are_averaged_before_cluster_bootstrap():
    rows = [r for d in (128, 1024, 4096, 8192)
            for i in range(3) for r in (row(d, i, i + .2), row(d, i, i - .2))]
    stats = causal_stats(rows)
    assert stats["n_trials"] == 24
    assert stats["n_pairs"] == 12
    assert stats["n_example_clusters"] == 3
    assert abs(stats["mean_gap_fraction"] - 1) < 1e-10
    assert stats["bootstrap_95_interval"][0] <= 1 <= stats["bootstrap_95_interval"][1]
