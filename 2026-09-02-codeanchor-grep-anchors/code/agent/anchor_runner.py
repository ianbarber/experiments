"""Launcher shim for arm E: run `mini-extra` with the `docker` environment mapped to
AnchorDockerEnvironment.

mini-swe-agent's SWE-bench runner only injects the per-instance image when
`environment_class` is literally "docker", so a dotted class path cannot be used from the
YAML. This shim rebinds the "docker" name to the anchor environment and then hands over
to the stock CLI unchanged:  python anchor_runner.py swebench --subset ... -c ...
"""

import sys

import minisweagent.environments as envs

envs._ENVIRONMENT_MAPPING["docker"] = "anchor_env.AnchorDockerEnvironment"

from minisweagent.run.utilities.mini_extra import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
