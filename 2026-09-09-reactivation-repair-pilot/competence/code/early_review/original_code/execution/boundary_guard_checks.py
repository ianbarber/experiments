"""CPU-only proof that an explicit administrative directory prevents induction."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts import program as p

assert p.INSTALLATIONS[0] == 1729
results = []
for recipe in p.RECIPES:
    for resume in (False, True):
        with tempfile.TemporaryDirectory(prefix='ledger-administrative-stop-proof-') as temporary:
            root = Path(temporary)
            target = root / 'checkpoints/induction_1729'
            target.mkdir(parents=True)
            marker = {'status': 'administrative_review_stop', 'reason': 'Synthetic boundary-guard fixture only'}
            (target / 'ADMINISTRATIVE_REVIEW_STOP.json').write_text(json.dumps(marker))
            runner = object.__new__(p.StageRunner)
            runner.root = root
            runner.results = root / 'results'
            runner.results.mkdir()
            runner.resume = resume
            calls = []
            runner.check = lambda: calls.append('source_check')
            runner.incoming_identity = lambda adapter: calls.append('incoming_identity')
            runner.event = lambda kind, **kwargs: calls.append(kind)
            stage = p.spec('induction_1729', 'train', 'data/induction.jsonl',
                           'checkpoints/induction_1729',
                           adapter=f'checkpoints/competence_{recipe}_i1729_epoch3/step_0096',
                           seed=1729, steps=96, save_steps=[32, 64, 96],
                           output_order=p.RECIPE_SETTINGS[recipe][1])
            with patch.object(p.subprocess, 'run', side_effect=AssertionError('No subprocess may run')) as run:
                try:
                    runner.stage(stage)
                    raise AssertionError('Administrative guard failed to stop induction')
                except ValueError as exc:
                    message = str(exc)
                assert run.call_count == 0
            assert calls == ['source_check', 'incoming_identity'], calls
            assert 'stage_start' not in calls and 'stage_reused' not in calls
            assert not (runner.results / 'logs').exists()
            assert not (target / 'COMPLETED.json').exists()
            assert set(x.name for x in target.iterdir()) == {'ADMINISTRATIVE_REVIEW_STOP.json'}
            if resume:
                assert 'Incomplete stage' in message
            else:
                assert 'Existing stage requires explicit' in message
            results.append({'recipe': recipe, 'resume': resume, 'blocked': True,
                            'subprocess_calls': 0, 'stage_start_events': 0, 'error': message})
print(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'status': 'passed',
                  'scope': 'Synthetic temporary directories only; original stage() and validate_contract() guard code, source/incoming checks stubbed as successful CPU preconditions',
                  'frozen_program_sha256': hashlib.sha256((ROOT / 'scripts/program.py').read_bytes()).hexdigest(),
                  'cases': results}, indent=2))
