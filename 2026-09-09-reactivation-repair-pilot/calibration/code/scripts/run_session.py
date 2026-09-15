"""Run the authorized preliminary check with an eight-hour GPU allocation cap."""
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from util import ROOT, now, event, notebook, write_json, check_freeze

CONFIG = json.loads((ROOT / 'configs/pilot.json').read_text())


def inspect(name):
    value = json.loads(subprocess.check_output(['docker','inspect',name], timeout=30))[0]
    return {'id':value['Id'], 'image':value['Image'], 'running':value['State']['Running']}


def interrupted(signum, frame):
    raise SystemExit(128 + signum)


def main():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, interrupted)
    results = ROOT / 'results'
    lock = (results / 'session.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (results / 'allocation.json').exists() or (results / 'TERMINAL.json').exists():
        raise RuntimeError('A prior allocation or terminal exists; no implicit reallocation or retry.')
    check_freeze()
    service = inspect(CONFIG['service_container'])
    research = inspect(CONFIG['research_container'])
    if service['id'] == research['id'] or not service['running'] or research['running']:
        raise RuntimeError('Expected original serving service running and distinct research container stopped.')
    if research['image'] != CONFIG['container_image']:
        raise RuntimeError('Research image differs from the frozen image.')
    started = time.time()
    state = {'started_at':now(), 'started_unix':started,
             'deadline_unix':started + CONFIG['gpu_budget_seconds'] - 180,
             'hard_allocation_deadline_unix':started + CONFIG['gpu_budget_seconds'],
             'shutdown_reserve_seconds':180,
             'budget_seconds':CONFIG['gpu_budget_seconds'],
             'service_id':service['id'], 'service_image':service['image'],
             'research_id':research['id'], 'research_image':research['image'],
             'authorization':'User authorized stopping SGLang and running the preliminary check on September 12, 2026.'}
    write_json(results / 'allocation.json', state)
    notebook('The reviewed inputs are frozen. The authorized eight-hour GPU allocation begins now, '
             'before stopping SGLang. An independent lease supervisor and the session wrapper both '
             'enforce its deadline and restore the exact original serving container.')
    process = supervisor = None
    code = None
    lease = results / 'service.lease'
    error = cleanup_error = None
    with (results / 'service_supervisor.log').open('x') as supervisor_log:
        try:
            supervisor = subprocess.Popen([sys.executable, str(ROOT / 'scripts/service_supervisor.py')],
                cwd=ROOT, stdout=supervisor_log, stderr=subprocess.STDOUT, start_new_session=True)
            stop_deadline = time.monotonic() + 100
            while inspect(service['id'])['running']:
                if lease.exists():
                    lease.touch()
                if supervisor.poll() is not None or time.monotonic() >= stop_deadline:
                    raise RuntimeError('Service pause did not finish within its bound.')
                time.sleep(2)
            lease.touch()
            subprocess.run(['docker','start',research['id']], check=True, timeout=40)
            lease.touch()
            process = subprocess.Popen([sys.executable, str(ROOT / 'scripts/program.py')], cwd=ROOT, start_new_session=True)
            while process.poll() is None:
                if time.time() >= state['deadline_unix']:
                    raise TimeoutError('Eight-hour allocated GPU budget exhausted.')
                if supervisor.poll() is not None or not lease.exists():
                    raise RuntimeError('Service supervisor/lease disappeared.')
                lease.touch()
                time.sleep(2)
            code = process.returncode
        except BaseException as exc:
            error = {'type':type(exc).__name__, 'message':str(exc)}
        finally:
            for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            try:
                if process is not None and process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait(timeout=10)
                # Release GPU work before waiting for the restoration supervisor.
                # Its independent deadline supplies a second stop path.
                subprocess.run(['docker','stop','--timeout','10',research['id']], check=True, timeout=45)
                if not inspect(research['id'])['running'] and not (results / 'allocation_released.json').exists():
                    write_json(results / 'allocation_released.json', {'at':now(), 'released_unix':time.time(), 'research_running':False})
                lease.unlink(missing_ok=True)
                if supervisor is not None:
                    try:
                        supervisor.wait(timeout=100)
                    except subprocess.TimeoutExpired:
                        supervisor.terminate()
                        supervisor.wait(timeout=100)
            except BaseException as exc:
                cleanup_error = {'type':type(exc).__name__, 'message':str(exc)}
            finally:
                try:
                    subprocess.run(['docker','stop','--timeout','10',research['id']], check=True, timeout=45)
                finally:
                    try:
                        if not inspect(research['id'])['running'] and not (results / 'allocation_released.json').exists():
                            write_json(results / 'allocation_released.json', {'at':now(), 'released_unix':time.time(), 'research_running':False})
                    finally:
                        subprocess.run(['docker','start',service['id']], check=True, timeout=90)
                        lease.unlink(missing_ok=True)
    if not (results / 'TERMINAL.json').exists():
        exhausted = time.time() >= state['deadline_unix']
        write_json(results / 'TERMINAL.json', {'at':now(), 'status':'early_failed' if exhausted else 'operational_error',
                   'failure':{'gate':'allocated_gpu_budget', 'shutdown_reserve_seconds':180} if exhausted else error,
                   'repair_training_launched':False, 'recorded_by':'session_wrapper_after_worker_stopped'})
    allocation_finished = json.loads((results / 'allocation_released.json').read_text())['released_unix']
    record = {'recorded_at':now(), 'service_id':service['id'], 'service_image':service['image'],
              'program_returncode':code, 'session_error':error, 'cleanup_error':cleanup_error,
              'research_allocation_elapsed_seconds':allocation_finished - started,
              'within_eight_hour_allocation':allocation_finished <= state['hard_allocation_deadline_unix'],
              'health_status':None, 'same_original_container_and_image':False}
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        current = inspect(service['id'])
        if current['image'] != service['image']:
            raise RuntimeError('Original service image identity changed during restoration.')
        try:
            with urllib.request.urlopen(CONFIG['service_health_url'], timeout=5) as response:
                if response.status == 200 and current['running']:
                    record.update(health_status=200, same_original_container_and_image=True,
                                  restored_at=now(), research_running=inspect(research['id'])['running'])
                    break
        except Exception:
            pass
        time.sleep(5)
    write_json(results / 'service_restoration.json', record)
    event('service_restoration_checked', health_status=record['health_status'],
          same_original_container_and_image=record['same_original_container_and_image'],
          research_allocation_elapsed_seconds=record['research_allocation_elapsed_seconds'])
    notebook(f'Original serving container restoration checked: HTTP {record["health_status"]}. '
             f'Research allocation used {record["research_allocation_elapsed_seconds"]:.1f} seconds. '
             'Raw operational identities are retained locally in the restoration record.')
    if record['health_status'] != 200 or record.get('research_running') is not False:
        raise RuntimeError('Restoration did not reach the required healthy/idle state.')
    if error or code != 0:
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
