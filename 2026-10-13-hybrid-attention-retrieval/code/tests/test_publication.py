import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location("publication",Path(__file__).parents[1]/"scripts"/"publication.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_locked_tests_use_each_seed_and_example_as_a_subject():
    rows=[]
    for seed in (42,314159):
        for example in range(3):
            for distance in (128,8192):
                for side in ("a","b"):
                    for component,layer,fraction in (("value_kv",24,.2),("recurrent_state",31,.01),("mlp",31,.25)):
                        rows.append({"seed":seed,"example":example,"distance":distance,"side":side,
                                     "component":component,"layer":layer,"control":0,"gap_fraction":fraction})
    result=module.locked_endpoints(rows)
    primary=result["global_kv_minus_recurrent_state_8192"]
    assert primary["n_examples"]==6
    assert abs(primary["mean"]-.19)<1e-12
    assert primary["decision"]=="support"
    assert result["layer31_mlp_transfer_8192"]["decision"]=="support"


def test_effect_threshold_can_reject_a_small_positive_mean():
    rows=[]
    for example in range(3):
        for component,layer,fraction in (("value_kv",24,.02),("recurrent_state",31,.01),("mlp",31,.03)):
            rows.append({"seed":42,"example":example,"distance":8192,"component":component,
                         "layer":layer,"control":0,"gap_fraction":fraction})
    result=module.locked_endpoints(rows)
    assert result["global_kv_minus_recurrent_state_8192"]["decision"]=="reject"
    assert result["layer31_mlp_transfer_8192"]["decision"]=="reject"
