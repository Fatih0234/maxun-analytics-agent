from __future__ import annotations

from analytics_agent.agent import graph as graph_module
from analytics_agent.engines.maxun.engine import MaxunWorkspaceEngine
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from tests.unit.test_maxun_engine import IDS


def test_maxun_graph_profile_has_no_chart_or_context_node(monkeypatch):
    monkeypatch.setattr(
        graph_module, "get_llm", lambda streaming=True: FakeListChatModel(responses=["done"])
    )
    engine = object.__new__(MaxunWorkspaceEngine)
    # The graph only needs the fixed tool surface when checking profile wiring.
    engine_tools = []
    graph = graph_module.build_graph(
        f"maxun:{IDS['workspace']}",
        engine=engine,
        engine_tools=engine_tools,
        context_tools=[],
        enabled_mutations=set(),
        maxun_readonly=True,
    )
    assert set(graph.get_graph().nodes) == {"__start__", "agent", "__end__"}
