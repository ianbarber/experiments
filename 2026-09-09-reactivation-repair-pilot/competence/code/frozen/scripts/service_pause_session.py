"""Pause the explicitly authorized local service, restoring it on exit or lost lease.

Operational helper only. The caller must already have permission to interrupt
the named service. Touch the lease while working; delete it to restore now.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--container', default='qwen3.8-27b-sglang')
p.add_argument('--lease', default='results/service_pause.lease')
p.add_argument('--lease-seconds', type=int, default=1200)
p.add_argument('--research-container', default='reactivation-competence-research')
a = p.parse_args()
lease = Path(a.lease)
log = lease.with_suffix('.jsonl')
lease.parent.mkdir(parents=True, exist_ok=True)

def record(event, **extra):
    value = {'at': datetime.now(timezone.utc).isoformat(), 'event': event, **extra}
    with log.open('a') as f:
        f.write(json.dumps(value) + '\n')
    print(json.dumps(value), flush=True)

def interrupted(signum, frame):
    raise SystemExit(f'Signal {signum}; restore service in finally')

for sig in [signal.SIGTERM, signal.SIGINT, signal.SIGHUP]:
    signal.signal(sig, interrupted)
info = json.loads(subprocess.check_output(['docker', 'inspect', a.container], timeout=30))[0]
identifier = info['Id']
research_info = json.loads(subprocess.check_output(['docker', 'inspect', a.research_container], timeout=30))[0]
research_identifier = research_info['Id']
if research_identifier == identifier:
    raise SystemExit('Serving and isolated research containers must differ.')
if not info['State']['Running']:
    raise SystemExit('Expected the authorized service to be running before pause.')
lease.touch()
record('authorized_pause_begin', container=a.container, container_id=identifier, image_id=info['Image'],
       lease_seconds=a.lease_seconds, authorization='User explicitly requested clearing the GPU and running this new follow-up through final review and publication on September 10, 2026; prior exact stop/restore authorization persists.')
try:
    subprocess.run(['docker', 'stop', '--time', '60', identifier], check=True, timeout=90)
    record('service_stopped', container_id=identifier)
    while lease.exists() and time.time() - lease.stat().st_mtime < a.lease_seconds:
        time.sleep(5)
    record('restore_requested', reason='lease removed' if not lease.exists() else 'lease expired')
finally:
    # Lease expiry can outlive the wrapper. Clear the exact isolated research
    # container captured at startup before returning memory to the serving model.
    try:
        subprocess.run(['docker', 'stop', '--timeout', '10', research_identifier], check=True, timeout=45)
        record('isolated_research_container_stopped', container_id=research_identifier)
    finally:
        subprocess.run(['docker', 'start', identifier], check=True, timeout=90)
        record('original_container_started', container_id=identifier)
        lease.unlink(missing_ok=True)
