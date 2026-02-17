"""Unit tests for output formatting node."""

from __future__ import annotations

from src.agent.nodes.format_output import format_output_node
from src.agent.state import RetrievedContext


def _ctx(*, source: str, doc_id: str, content: str = "x") -> RetrievedContext:
    return RetrievedContext(
        content=content,
        source=source,
        doc_id=doc_id,
        relevance_score=0.9,
        chunk_index=0,
    )


def test_deduplicates_sources_by_source_path_not_chunk_doc_id() -> None:
    state = {
        "draft_answer": "Attention is a weighted aggregation mechanism.",
        "retrieved_contexts": [
            _ctx(source="C:/docs/1706.03762v7.pdf", doc_id="doc-1-chunk-0"),
            _ctx(source="C:/docs/1706.03762v7.pdf", doc_id="doc-1-chunk-1"),
            _ctx(source="C:/docs/1706.03762v7.pdf", doc_id="doc-1-chunk-2"),
        ],
    }

    out = format_output_node(state)
    final = out["final_answer"]

    assert "**Sources**:" in final
    assert final.count("- C:/docs/1706.03762v7.pdf") == 1
    assert out["reasoning_trace"][0].metadata["num_sources"] == 3
    assert out["reasoning_trace"][0].metadata["num_unique_sources"] == 1


def test_keeps_distinct_sources() -> None:
    state = {
        "draft_answer": "Transformers use self-attention and feed-forward blocks.",
        "retrieved_contexts": [
            _ctx(source="paper-A.pdf", doc_id="a-0"),
            _ctx(source="paper-B.pdf", doc_id="b-0"),
        ],
    }

    out = format_output_node(state)
    final = out["final_answer"]

    assert final.count("- paper-A.pdf") == 1
    assert final.count("- paper-B.pdf") == 1
    assert out["reasoning_trace"][0].metadata["num_unique_sources"] == 2
