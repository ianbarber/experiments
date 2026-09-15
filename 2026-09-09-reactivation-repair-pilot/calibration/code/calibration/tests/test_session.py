"""Mocked CPU checks of the allocation and original-service restoration contract.

No subprocess, Docker, GPU, HTTP, signal handler, or real sleep is invoked.
"""

from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

SOURCE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE_ROOT / "scripts"))
import run_session
import service_supervisor


SERVICE = "original-service-full-container-id"
RESEARCH = "new-research-full-container-id"
SERVICE_IMAGE = "original-service-frozen-image"
RESEARCH_IMAGE = "new-research-frozen-image"


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class Clock:
    def __init__(self):
        self.start = time.time()
        self.current = self.start
        self.on_sleep = None

    def sleep(self, seconds):
        self.current += seconds
        if self.on_sleep:
            self.on_sleep()

    def namespace(self):
        return SimpleNamespace(time=lambda: self.current, monotonic=lambda: self.current, sleep=self.sleep)


class Child:
    def __init__(self, polls=None, on_complete=None, on_wait=None):
        self.polls = iter(polls) if polls is not None else None
        self.returncode = None
        self.pid = 12345
        self.on_complete = on_complete
        self.on_wait = on_wait

    def poll(self):
        if self.returncode is not None:
            return self.returncode
        if self.polls is not None:
            value = next(self.polls, 0)
            if value is not None:
                self.returncode = value
                if self.on_complete:
                    self.on_complete()
        return self.returncode

    def wait(self, timeout=None):
        if self.on_wait:
            self.on_wait()
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self):
        self.returncode = -signal.SIGTERM


@contextmanager
def wrapper_harness(program_launch_error=False, program_forever=False, program_code=0):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        (root / "results").mkdir()
        clock = Clock()
        config = dict(run_session.CONFIG, service_container="service-name", research_container="research-name", container_image=RESEARCH_IMAGE, gpu_budget_seconds=300)
        states = {
            SERVICE: dict(id=SERVICE, image=SERVICE_IMAGE, running=True),
            RESEARCH: dict(id=RESEARCH, image=RESEARCH_IMAGE, running=False),
        }
        commands = []
        actions = []
        lease_mtimes = []
        lease_touches = []
        original_touch = Path.touch
        def record_touch(path, *args, **kwargs):
            if path == root / "results/service.lease":
                lease_touches.append(clock.current)
            return original_touch(path, *args, **kwargs)
        stack.enter_context(patch.object(Path, "touch", autospec=True, side_effect=record_touch))
        def inspect(name):
            identity = {"service-name": SERVICE, "research-name": RESEARCH}.get(name, name)
            return dict(states[identity])
        inspect_mock = Mock(side_effect=inspect)
        def run(command, **kwargs):
            commands.append(command)
            actions.append(("subprocess", command))
            if command[:2] == ["docker", "start"]:
                states[command[-1]]["running"] = True
            elif command[:2] == ["docker", "stop"]:
                states[command[-1]]["running"] = False
            else:
                raise AssertionError(f"Unexpected subprocess: {command}")
            return SimpleNamespace(returncode=0)
        supervisor = Child(on_wait=lambda: actions.append(("supervisor_wait", None)))
        def finished():
            dump(root / "results/TERMINAL.json", {"status": "passed" if program_code == 0 else "operational_error"})
        program = Child(None if program_forever else [None, None, program_code], on_complete=finished)
        def popen(command, **kwargs):
            filename = Path(command[1]).name
            if filename == "service_supervisor.py":
                states[SERVICE]["running"] = False
                (root / "results/service.lease").touch()
                return supervisor
            if filename == "program.py":
                if program_launch_error:
                    raise OSError("Synthetic program launch failure")
                return program
            raise AssertionError(f"Unexpected Popen: {command}")
        def killpg(pid, signum):
            self_pid = program.pid
            if pid != self_pid:
                raise AssertionError("Only the launched worker process group may be signalled")
            program.returncode = -signum
        health = Mock()
        health.__enter__ = Mock(return_value=SimpleNamespace(status=200))
        health.__exit__ = Mock(return_value=False)
        def observed_sleep():
            lease = root / "results/service.lease"
            if lease.exists():
                lease_mtimes.append(lease.stat().st_mtime)
        clock.on_sleep = observed_sleep
        replacements = {
            "ROOT": root, "CONFIG": config, "inspect": inspect_mock,
            "check_freeze": Mock(), "now": lambda: "synthetic-test-time",
            "event": Mock(), "notebook": Mock(), "write_json": dump,
            "time": clock.namespace(),
            "signal": SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIGHUP=signal.SIGHUP, SIGKILL=signal.SIGKILL, SIG_IGN=signal.SIG_IGN, signal=Mock()),
            "os": SimpleNamespace(killpg=Mock(side_effect=killpg)),
            "subprocess": SimpleNamespace(Popen=Mock(side_effect=popen), run=Mock(side_effect=run), STDOUT=subprocess.STDOUT, TimeoutExpired=subprocess.TimeoutExpired),
            "urllib": SimpleNamespace(request=SimpleNamespace(urlopen=Mock(return_value=health))),
        }
        for name, value in replacements.items():
            stack.enter_context(patch.object(run_session, name, value))
        yield SimpleNamespace(root=root, clock=clock, config=config, states=states, commands=commands, actions=actions, program=program, supervisor=supervisor, inspect=inspect_mock, lease_mtimes=lease_mtimes, lease_touches=lease_touches)


class SessionWrapper(unittest.TestCase):
    def assert_original_restored(self, h):
        stops = [c for c in h.commands if c[:2] == ["docker", "stop"]]
        starts = [c for c in h.commands if c[:2] == ["docker", "start"]]
        self.assertTrue(stops)
        self.assertTrue(all(c[-1] == RESEARCH for c in stops))
        self.assertEqual(starts[-1], ["docker", "start", SERVICE])
        stop_position = next(i for i, action in enumerate(h.actions) if action[0] == "subprocess" and action[1][:2] == ["docker", "stop"] and action[1][-1] == RESEARCH)
        wait_position = next(i for i, action in enumerate(h.actions) if action[0] == "supervisor_wait")
        self.assertLess(stop_position, wait_position, "Research must stop before waiting for the supervisor")
        self.assertFalse(h.states[RESEARCH]["running"])
        self.assertTrue(h.states[SERVICE]["running"])
        self.assertFalse((h.root / "results/service.lease").exists())
        receipt = json.loads((h.root / "results/service_restoration.json").read_text())
        self.assertEqual(receipt["service_id"], SERVICE)
        self.assertEqual(receipt["service_image"], SERVICE_IMAGE)
        self.assertTrue(receipt["same_original_container_and_image"])
        self.assertEqual(receipt["health_status"], 200)
        self.assertFalse(receipt["research_running"])

    def test_normal_completion_stops_research_and_restores_original_identity(self):
        with wrapper_harness() as h:
            run_session.main()
            self.assert_original_restored(h)
            state = json.loads((h.root / "results/allocation.json").read_text())
            self.assertEqual(state["hard_allocation_deadline_unix"] - state["started_unix"], 300)
            self.assertEqual(state["hard_allocation_deadline_unix"] - state["deadline_unix"], 180)
            self.assertEqual(state["shutdown_reserve_seconds"], 180)
            self.assertGreaterEqual(len(h.lease_mtimes), 2)
            self.assertTrue(any(t > h.clock.start for t in h.lease_touches), "The wrapper must renew the lease as the program runs")
            self.assertEqual(json.loads((h.root / "results/TERMINAL.json").read_text())["status"], "passed")

    def test_program_launch_error_still_restores_original_identity(self):
        with wrapper_harness(program_launch_error=True) as h:
            with self.assertRaises(SystemExit):
                run_session.main()
            self.assert_original_restored(h)
            terminal = json.loads((h.root / "results/TERMINAL.json").read_text())
            self.assertEqual(terminal["status"], "operational_error")
            self.assertEqual(terminal["failure"]["type"], "OSError")
            self.assertFalse(terminal["repair_training_launched"])

    def test_worker_nonzero_exit_is_propagated_after_original_restoration(self):
        with wrapper_harness(program_code=17) as h:
            with self.assertRaises(SystemExit) as raised:
                run_session.main()
            self.assertEqual(raised.exception.code, 17)
            self.assert_original_restored(h)

    def test_deadline_stops_running_worker_with_shutdown_reserve(self):
        with wrapper_harness(program_forever=True) as h:
            with self.assertRaises(SystemExit):
                run_session.main()
            self.assert_original_restored(h)
            terminal = json.loads((h.root / "results/TERMINAL.json").read_text())
            self.assertEqual(terminal["status"], "resource_limited_incomplete")
            self.assertEqual(terminal["failure"]["gate"], "allocated_gpu_budget")
            run_session.os.killpg.assert_called_once_with(h.program.pid, signal.SIGTERM)
            state = json.loads((h.root / "results/allocation.json").read_text())
            self.assertEqual(h.clock.current, state["deadline_unix"])
            self.assertLess(h.clock.current, state["hard_allocation_deadline_unix"])

    def test_prior_allocation_or_terminal_prevents_any_service_action(self):
        for existing in ("allocation.json", "TERMINAL.json"):
            with self.subTest(existing=existing), wrapper_harness() as h:
                dump(h.root / "results" / existing, {})
                with self.assertRaisesRegex(RuntimeError, "no implicit reallocation"):
                    run_session.main()
                h.inspect.assert_not_called()
                run_session.subprocess.Popen.assert_not_called()
                run_session.subprocess.run.assert_not_called()


@contextmanager
def supervisor_harness(mode="unlink", fail_initial_stop=False, fail_research_stop=False):
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        root = Path(directory)
        (root / "results").mkdir()
        clock = Clock()
        deadline = clock.current + (10 if mode == "deadline" else 1000)
        dump(root / "results/allocation.json", {"service_id": SERVICE, "research_id": RESEARCH, "deadline_unix": deadline})
        commands = []
        def run(command, **kwargs):
            commands.append(command)
            if fail_initial_stop and command[-1] == SERVICE and command[1] == "stop":
                raise subprocess.CalledProcessError(1, command)
            if fail_research_stop and command[-1] == RESEARCH and command[1] == "stop":
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(returncode=0)
        def on_sleep():
            lease = root / "results/service.lease"
            if mode == "unlink":
                lease.unlink(missing_ok=True)
            elif mode == "deadline" and lease.exists():
                # Simulate a live wrapper renewing the lease while time advances.
                os.utime(lease, (clock.current, clock.current))
        clock.on_sleep = on_sleep
        replacements = {
            "ROOT": root, "now": lambda: "synthetic-test-time", "event": Mock(), "write_json": dump,
            "time": clock.namespace(),
            "signal": SimpleNamespace(SIGTERM=signal.SIGTERM, SIGINT=signal.SIGINT, SIGHUP=signal.SIGHUP, SIG_IGN=signal.SIG_IGN, signal=Mock()),
            "subprocess": SimpleNamespace(run=Mock(side_effect=run), check_output=Mock(return_value=json.dumps([{"State": {"Running": False}}]).encode())),
        }
        for name, value in replacements.items():
            stack.enter_context(patch.object(service_supervisor, name, value))
        yield SimpleNamespace(root=root, clock=clock, commands=commands, deadline=deadline)


class IndependentLeaseSupervisor(unittest.TestCase):
    def assert_exact_cleanup(self, h):
        self.assertEqual(h.commands, [
            ["docker", "stop", "--timeout", "60", SERVICE],
            ["docker", "stop", "--timeout", "10", RESEARCH],
            ["docker", "start", SERVICE],
        ])
        self.assertFalse((h.root / "results/service.lease").exists())
        receipt = json.loads((h.root / "results/allocation_released.json").read_text())
        self.assertFalse(receipt["research_running"])

    def test_lease_removal_restores_service(self):
        with supervisor_harness() as h:
            service_supervisor.main()
            self.assert_exact_cleanup(h)
            self.assertEqual(h.clock.current - h.clock.start, 2)

    def test_healthy_heartbeat_cannot_extend_allocation_deadline(self):
        with supervisor_harness(mode="deadline") as h:
            service_supervisor.main()
            self.assert_exact_cleanup(h)
            self.assertEqual(h.clock.current, h.deadline)

    def test_stale_heartbeat_restores_service_before_long_deadline(self):
        with supervisor_harness(mode="stale") as h:
            service_supervisor.main()
            self.assert_exact_cleanup(h)
            elapsed = h.clock.current - h.clock.start
            self.assertGreaterEqual(elapsed, 120)
            self.assertLessEqual(elapsed, 122)
            self.assertLess(h.clock.current, h.deadline)

    def test_service_pause_failure_still_attempts_exact_cleanup(self):
        with supervisor_harness(fail_initial_stop=True) as h:
            with self.assertRaises(subprocess.CalledProcessError):
                service_supervisor.main()
            self.assert_exact_cleanup(h)

    def test_research_stop_failure_still_attempts_original_start(self):
        with supervisor_harness(fail_research_stop=True) as h:
            with self.assertRaises(subprocess.CalledProcessError):
                service_supervisor.main()
            self.assertEqual(h.commands[-1], ["docker", "start", SERVICE])
            self.assertFalse((h.root / "results/service.lease").exists())
            self.assertFalse((h.root / "results/allocation_released.json").exists())


if __name__ == "__main__":
    unittest.main()
