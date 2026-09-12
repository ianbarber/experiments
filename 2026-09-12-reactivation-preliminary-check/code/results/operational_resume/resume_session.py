"""Retain the frozen session/supervisor lifecycle with the remaining GPU budget."""
import argparse
import json
import sys

from resume_common import ROOT, HERE, verify_plan


class ProgramSubprocessProxy:
    """Only replace the frozen worker launch; all lifecycle operations delegate."""
    def __init__(self, original, executable=sys.executable, root=ROOT, here=HERE):
        self.original = original
        self.original_worker = [executable, str(root / 'scripts/program.py')]
        self.resume_worker = [executable, str(here / 'resume_program.py')]

    def Popen(self, command, *args, **kwargs):
        if command == self.original_worker:
            command = self.resume_worker
        return self.original.Popen(command, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.original, name)


def configure_session(session, plan):
    session.CONFIG = dict(session.CONFIG, gpu_budget_seconds=plan['budget']['remaining_budget_seconds'])
    session.subprocess = ProgramSubprocessProxy(session.subprocess)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan, config = verify_plan()
    if args.check_only:
        print(json.dumps(dict(status='check_passed', scope='No GPU/model/service calls',
                              budget=plan['budget'], frozen_config_file_unchanged=True,
                              lifecycle='Original session and independent supervisor retained')), flush=True)
        return
    # This is a one-shot continuation. The frozen session itself independently
    # refuses an extant allocation/terminal before performing container calls.
    sys.path.insert(0, str(ROOT / 'scripts'))
    import run_session
    configure_session(run_session, plan)
    run_session.main()


if __name__ == '__main__':
    main()
