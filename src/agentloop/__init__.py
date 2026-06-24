"""agentloop -- a generic multi-agent loop over CLI coding agents."""

from __future__ import annotations

from .adapters.claude import ClaudeAgent
from .adapters.codex import CodexAgent
from .agent import Agent, AgentError, CliAgent
from .domain import Message, Transcript, TurnResult
from .orchestrator import LoopResult, Orchestrator, fan_out
from .policy import Context, DebatePolicy, Policy, RoundRobinPolicy
from .stop import BudgetUSD, Consensus, MaxRounds, StopCondition
from .store import JournalStore, Store

__all__ = [
    "Agent",
    "AgentError",
    "BudgetUSD",
    "ClaudeAgent",
    "CliAgent",
    "CodexAgent",
    "Consensus",
    "Context",
    "DebatePolicy",
    "JournalStore",
    "LoopResult",
    "MaxRounds",
    "Message",
    "Orchestrator",
    "Policy",
    "RoundRobinPolicy",
    "StopCondition",
    "Store",
    "Transcript",
    "TurnResult",
    "fan_out",
]
