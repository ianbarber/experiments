"""Replay the frozen controller, reusing only its two verified completed stages."""
import argparse
import json
from pathlib import Path
import sys

from resume_common import ROOT, HERE, make_plan, verify_plan, verify_stage


def stage_wrapper(original, program, plan, config):
    reused = set()

    def stage(name, mode, data, seed, adapter=None, epoch=1, optimizer=None, sample=False):
        program.budget_check()
        program.check_freeze()
        output = ROOT / 'results/stages' / name
        if name in plan['reusable_stages']:
            if name in reused:
                raise ValueError(f'Controller requested a completed stage more than once: {name}')
            verify_stage(ROOT, config, name, mode, data, seed, adapter, epoch, optimizer, sample,
                         expected_completion_sha256=plan['reusable_stages'][name]['completion_sha256'])
            reused.add(name)
            program.event('operational_resume_reused_stage', stage=name,
                          completion_sha256=plan['reusable_stages'][name]['completion_sha256'])
            return output
        if output.exists():
            raise ValueError(f'Existing unbound/incomplete stage cannot be retried or reused: {name}')
        return original(name, mode, data, seed, adapter, epoch, optimizer, sample)

    return stage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--freeze-plan', action='store_true')
    parser.add_argument('--prior-attempt', type=Path)
    args = parser.parse_args()
    if args.freeze_plan:
        if args.prior_attempt is None or args.check_only:
            parser.error('--freeze-plan requires --prior-attempt and cannot combine with --check-only.')
        directory = args.prior_attempt if args.prior_attempt.is_absolute() else ROOT / args.prior_attempt
        plan = make_plan(directory)
        with (HERE / 'RESUME_FREEZE.json').open('x') as stream:
            stream.write(json.dumps(plan, indent=2) + '\n')
        print(json.dumps(dict(status='recovery_frozen', budget=plan['budget'])), flush=True)
        return
    if args.prior_attempt is not None:
        parser.error('--prior-attempt is only used when freezing the recovery plan.')
    plan, config = verify_plan()
    if args.check_only:
        print(json.dumps(dict(status='check_passed', scope='No GPU/model/service calls',
                              reusable_stages=list(plan['reusable_stages']), budget=plan['budget'])), flush=True)
        return
    sys.path.insert(0, str(ROOT / 'scripts'))
    import program
    program.stage = stage_wrapper(program.stage, program, plan, config)
    program.event('explicit_operational_resume', recovery_freeze='results/operational_resume/RESUME_FREEZE.json',
                  preserved_stages=list(plan['reusable_stages']), no_model_or_gate_changes=True,
                  no_new_budget=True, prior_charged_seconds=plan['budget']['prior_charged_seconds'])
    program.main()


if __name__ == '__main__':
    main()
