"""Run the authorized follow-up while preserving and restoring the serving container."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
SERVICE = 'qwen3.8-27b-sglang'
RESEARCH = 'reactivation-competence-research'
lease = ROOT / 'results/service_pause.lease'


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    if (ROOT / 'results/PROGRAM_COMPLETED.json').exists():
        raise RuntimeError('The fixed program is already complete; do not pause service to rerun it.')
    origin_path = ROOT / 'results/SESSION_ORIGIN.json'
    if args.resume:
        origin = json.loads(origin_path.read_text())
    else:
        origin = {'started_epoch': time.time(), 'started_utc': datetime.now(timezone.utc).isoformat(),
                  'ceiling_seconds': 172800, 'scope': 'Cumulative wall time including intervals between resumed sessions.'}
        with origin_path.open('x') as stream:
            stream.write(json.dumps(origin, indent=2) + '\n')
    if time.time() - origin['started_epoch'] > origin['ceiling_seconds']:
        raise TimeoutError('The cumulative 48-hour operational ceiling already expired.')
    research_before = json.loads(subprocess.check_output(['docker', 'inspect', RESEARCH], timeout=30))[0]
    research_identifier = research_before['Id']
    before = json.loads(subprocess.check_output(['docker', 'inspect', SERVICE], timeout=30))[0]
    if research_identifier == before['Id']:
        raise RuntimeError('Serving and research containers must differ.')
    if not before['State']['Running']:
        raise RuntimeError('Expected the serving container to be running before the authorized pause.')
    supervisor = process = None
    code = None
    log = (ROOT / ('results/service_supervisor_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.log')).open('x')
    try:
        supervisor = subprocess.Popen([sys.executable, str(ROOT / 'scripts/service_pause_session.py'),
                         '--container', SERVICE, '--research-container', RESEARCH, '--lease', str(lease)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 100
        while time.monotonic() < deadline:
            if supervisor.poll() is not None:
                raise RuntimeError('Service-pause supervisor exited unexpectedly.')
            state = json.loads(subprocess.check_output(['docker', 'inspect', before['Id']], timeout=30))[0]
            if not state['State']['Running']:
                break
            time.sleep(2)
        else:
            raise TimeoutError('Service did not stop within the bounded startup wait.')
        subprocess.run(['docker', 'start', research_identifier], check=True, timeout=40)
        process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/program.py')] + (['--resume'] if args.resume else []), cwd=ROOT, start_new_session=True)
        started = time.monotonic()
        while True:
            if time.time() - origin['started_epoch'] > origin['ceiling_seconds']:
                raise TimeoutError('Forty-eight-hour operational ceiling exceeded; preserve partial records.')
            if supervisor.poll() is not None or not lease.exists():
                raise RuntimeError('Service-pause lease or supervisor disappeared.')
            lease.touch()
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                pass
    finally:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, signal.SIG_IGN)
        try:
            if process is not None and (process.poll() is None or code != 0):
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                # A failed docker-exec client can leave its remote worker alive.
                # Clear only our isolated research container before restoration.
                subprocess.run(['docker', 'stop', '--timeout', '10', research_identifier], check=True, timeout=40)
                subprocess.run(['docker', 'start', research_identifier], check=True, timeout=40)
        finally:
            lease.unlink(missing_ok=True)
            cleanup_error = None
            try:
                if supervisor is not None:
                    try:
                        supervisor.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        supervisor.terminate()
                        supervisor.wait(timeout=90)
            except Exception as exc:
                cleanup_error = repr(exc)
            finally:
                # The last-resort original-container start must still execute
                # if waiting for or signalling the supervisor itself fails.
                try:
                    subprocess.run(['docker', 'stop', '--timeout', '10', research_identifier], check=True, timeout=45)
                finally:
                    subprocess.run(['docker', 'start', before['Id']], check=True, timeout=90)
                    log.close()
            record = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'container_id': before['Id'],
                      'image_id': before['Image'], 'program_returncode': code, 'health_status': None,
                      'supervisor_cleanup_error': cleanup_error}
            deadline = time.monotonic() + 900
            while time.monotonic() < deadline:
                after = json.loads(subprocess.check_output(['docker', 'inspect', before['Id']], timeout=30))[0]
                if after['Image'] != before['Image']:
                    raise RuntimeError('Restored service image identity changed.')
                try:
                    with urllib.request.urlopen('http://localhost:8888/health', timeout=5) as response:
                        if response.status == 200 and after['State']['Running']:
                            record.update(checked_utc=datetime.now(timezone.utc).isoformat(), health_status=200,
                                          same_original_container_and_image=True, running=after['State']['Running'])
                            break
                except Exception:
                    pass
                time.sleep(10)
            (ROOT / 'results/service_restoration.json').write_text(json.dumps(record, indent=2) + '\n')
            if record['health_status'] != 200:
                raise RuntimeError('Original service started but health did not reach 200 within 15 minutes.')
    if code != 0:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
