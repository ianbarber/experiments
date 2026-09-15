"""CPU-only proof of the frozen driver's existing-stage boundary rejection.

Uses only temporary synthetic output directories. Source/incoming identity calls
are stubs; this tests rejection ordering, not checkpoint identity validation.
Never invokes Docker, models, GPU, or the running controller.
"""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
DEST = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    freeze = json.loads((ROOT / 'results/FROZEN_PROGRAM.json').read_text())
    archived = ROOT / freeze['source_snapshot'] / 'scripts'
    for relative, expected in freeze['pins'].items():
        assert sha(ROOT / relative) == expected, relative
        if not relative.startswith('data/'):
            assert sha(ROOT / freeze['source_snapshot'] / relative) == expected, relative
    sys.path.insert(0, str(archived))
    spec = importlib.util.spec_from_file_location('frozen_boundary_program', archived / 'program.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.INSTALLATIONS[0] == 1729
    records = []
    for resume in (False, True):
        with tempfile.TemporaryDirectory(prefix='boundary-guard-cpu-') as temporary:
            root = Path(temporary)
            output = root / 'checkpoints/induction_1729'
            output.mkdir(parents=True, exist_ok=False)
            marker = output / 'ADMINISTRATIVE_STOP.json'
            marker.write_text('{"synthetic":true,"not_a_model_checkpoint":true}\n')
            original_marker = marker.read_bytes()
            runner = object.__new__(module.StageRunner)
            runner.root = root
            runner.results = root / 'results'
            runner.results.mkdir()
            runner.resume = resume
            runner.stages = []
            calls = []
            runner.check = lambda: calls.append('source_check')
            runner.incoming_identity = lambda adapter: calls.append('incoming_identity')
            runner.event = lambda *args, **kwargs: calls.append('event')
            stage = module.spec('induction_1729', 'train', 'data/induction.jsonl',
                'checkpoints/induction_1729', adapter='checkpoints/fixture/step_0096',
                seed=1729, steps=96, save_steps=[32,64,96], output_order='decision_first')
            with patch.object(module.subprocess, 'run', side_effect=AssertionError('A subprocess must never run')) as run:
                try:
                    runner.stage(stage)
                except ValueError as exc:
                    error = str(exc)
                else:
                    raise AssertionError('Guard failed to reject administrative marker')
                run.assert_not_called()
            expected = ('Incomplete stage:' if resume else 'Existing stage requires explicit --resume')
            assert error.startswith(expected), error
            assert calls == ['source_check', 'incoming_identity'], calls
            assert marker.read_bytes() == original_marker
            assert not (output / 'COMPLETED.json').exists()
            assert not (runner.results / 'logs').exists()
            assert runner.stages == []
            records.append(dict(resume=resume,error=error,source_and_incoming_checks_precede_guard=True,
                stage_start_events=0,subprocess_calls=0,created_completion=False,
                created_log=False,marker_unchanged=True))
    assert not any(name in sys.modules for name in ('torch','transformers','peft'))
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),status='passed',
        script_sha256=sha(Path(__file__)),frozen_program_sha256=sha(ROOT/'results/FROZEN_PROGRAM.json'),
        program_sha256=sha(archived/'program.py'),cases=records,
        model_gpu_service_calls=0,actual_checkpoint_paths_mutated=False,
        scope='Focused synthetic guard-order proof only; no live runner/source-check/weight/service mock integration.')
    (DEST/'BOUNDARY_GUARD_CPU_CHECK.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
