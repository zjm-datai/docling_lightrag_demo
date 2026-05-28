"""Tests for the unified retrieval adapter."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

from sra_agent.retrieval_tool import UnifiedRetrievalTool


class SlowCountingRetriever:
    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def retrieve(self, query: str, top_k: int = 5, mode: str = "hybrid"):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)

        time.sleep(0.05)

        with self.lock:
            self.active -= 1

        return [
            {
                "content": f"result for {query}",
                "metadata": {"mode": mode, "retriever": "fake"},
                "score": 1.0,
            }
        ]


def test_unified_retrieval_serializes_backend_calls():
    retriever = SlowCountingRetriever()
    tool = UnifiedRetrievalTool(retriever)

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(
            executor.map(
                lambda index: tool.retrieve(
                    f"q{index}",
                    purpose="test",
                    top_k=1,
                    mode="hybrid",
                ),
                range(3),
            )
        )

    assert retriever.max_active == 1
    assert [item[0].content for item in results] == [
        "result for q0",
        "result for q1",
        "result for q2",
    ]
