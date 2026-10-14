import re

import pytest
from transformers import AutoTokenizer

from fwlayer.data import answer_correct, first_difference, make_pair
from fwlayer.run import MODEL, REVISION


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL, revision=REVISION)


def test_first_difference_handles_a_shared_value_prefix():
    assert first_difference([85, 22, 18, 15], [85, 22, 22, 20]) == 2
    with pytest.raises(ValueError):
        first_difference([85, 22], [85, 22, 20])


@pytest.mark.parametrize("distance,index", [(128, 0), (1024, 1), (8192, 2)])
def test_pairs_preserve_value_identity_positions_and_oracle_labels(tokenizer, distance, index):
    pair = make_pair(tokenizer, distance, index)
    assert len(pair.ids_a) == len(pair.ids_b)
    assert pair.token_a != pair.token_b
    assert pair.answer_ids_a[:pair.differing_index] == pair.answer_ids_b[:pair.differing_index]
    assert pair.ids_a[-len(pair.shared_answer_ids):] == pair.shared_answer_ids
    assert max(pair.distance_a, pair.distance_b) == distance
    for span in pair.value_positions:
        assert [pair.ids_a[i] for i in span] == [pair.ids_b[i] for i in span]
    for text, expected in ((pair.prompt_a, pair.answer_a), (pair.prompt_b, pair.answer_b)):
        facts = dict(re.findall(r"(e[0-9]{3})'s a0 is (v[0-9]{3})\.", text))
        assert len(facts) == pair.distractors + 2
        assert facts[pair.query_key] == expected
    assert pair.prompt_a[-50:] == pair.prompt_b[-50:]
    assert set(pair.changed_positions).isdisjoint({p for span in pair.value_positions for p in span})


def test_distance_changes_do_not_change_the_fact_inventory(tokenizer):
    short, long = make_pair(tokenizer, 128, 3), make_pair(tokenizer, 4096, 3)
    assert (short.answer_a, short.answer_b, short.query_key) == (long.answer_a, long.answer_b, long.query_key)
    assert short.source_positions == long.source_positions
    assert short.changed_positions == long.changed_positions
    assert long.query_position - short.query_position == 4096 - 128


@pytest.mark.parametrize("prediction,correct", [("v123\n", True), ("v123.", True),
                                                ("v1234", False), ("v124", False),
                                                ("The value is v123", False)])
def test_full_answer_accuracy_does_not_accept_a_prefix_collision(prediction, correct):
    assert answer_correct(prediction, "v123") == correct
