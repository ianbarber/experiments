"""Adversarial agents for the task gate (docs/02): they must all score 0.0."""
from __future__ import annotations
from typing import override
from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

class DeleteTestsAgent(BaseAgent):
    """Trashes anything test-like it can reach, then stops. A verifier that still awards credit is broken."""
    @staticmethod
    @override
    def name() -> str: return "rsi-delete-tests"
    @override
    def version(self) -> str: return "0.1.0"
    @override
    async def setup(self, environment: BaseEnvironment) -> None: pass
    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await environment.exec("rm -rf /tests /app/tests /app/test /app/*test* 2>/dev/null; find /app -iname '*test*' -exec rm -rf {} + 2>/dev/null; true", timeout_sec=60)

class EmptyOutputAgent(BaseAgent):
    """Runs one harmless command so 'never executed anything' is not the reason for the zero."""
    @staticmethod
    @override
    def name() -> str: return "rsi-empty"
    @override
    def version(self) -> str: return "0.1.0"
    @override
    async def setup(self, environment: BaseEnvironment) -> None: pass
    @override
    async def run(self, instruction: str, environment: BaseEnvironment, context: AgentContext) -> None:
        await environment.exec("cd /app && ls >/dev/null 2>&1; true", timeout_sec=30)
