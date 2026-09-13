"""Independent bounded lease: stop only the authorized service and restore it."""
import json
from pathlib import Path
import signal
import subprocess
import time
from util import ROOT, event, now, write_json


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    state = json.loads((ROOT / 'results/allocation.json').read_text())
    lease = ROOT / 'results/service.lease'
    service, research = state['service_id'], state['research_id']
    if service == research:
        raise RuntimeError('Research and serving identity must differ.')
    lease.touch()
    try:
        event('authorized_service_pause', allocation_deadline_unix=state['deadline_unix'])
        subprocess.run(['docker','stop','--timeout','60',service], check=True, timeout=90)
        event('original_service_stopped')
        while lease.exists() and time.time() - lease.stat().st_mtime < 120 and time.time() < state['deadline_unix']:
            time.sleep(2)
        event('service_restore_requested', lease_exists=lease.exists(), budget_expired=time.time() >= state['deadline_unix'])
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            subprocess.run(['docker','stop','--timeout','10',research], check=True, timeout=45)
            running = json.loads(subprocess.check_output(['docker','inspect',research], timeout=30))[0]['State']['Running']
            if not running:
                write_json(ROOT / 'results/allocation_released.json', {'at':now(), 'released_unix':time.time(), 'research_running':False})
        finally:
            subprocess.run(['docker','start',service], check=True, timeout=90)
            lease.unlink(missing_ok=True)
            event('original_service_start_requested')


if __name__ == '__main__':
    main()
