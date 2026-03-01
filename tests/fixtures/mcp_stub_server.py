"""Minimal MCP server stub over stdio for integration tests (H9).

Reads JSON-RPC 2.0 messages from stdin (one JSON object per line) and writes
responses to stdout. Handles initialize, tools/list, and tools/call for
query_knowledge_hub so that MCPClientManager can connect and RAGToolWrapper
can parse the result. Stderr is left unchanged so test logs are visible.
"""

from __future__ import annotations

import json
import sys


def main() -> None:
    # Fixed tool call result that RAGToolWrapper._parse_search_results accepts
    search_result = {
        "results": [
            {
                "content": "Stub context for MCP subprocess test.",
                "source": "MCP Stub",
                "doc_id": "stub-doc-001",
                "relevance_score": 0.95,
                "chunk_index": 0,
            }
        ]
    }
    search_result_text = json.dumps(search_result)

    tool_def = {
        "name": "query_knowledge_hub",
        "description": "Query the knowledge hub (stub).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "collection_name": {"type": "string"},
                "top_k": {"type": "integer"},
            },
        },
    }

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        req_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params") or {}

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "mcp-stub", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {"tools": [tool_def]}
        elif method == "tools/call":
            name = params.get("name", "")
            if name == "query_knowledge_hub":
                result = {
                    "content": [{"type": "text", "text": search_result_text}],
                    "isError": False,
                }
            else:
                result = {"content": [{"type": "text", "text": "{}"}], "isError": False}
        else:
            result = {}

        if req_id is None:
            continue
        out = {"jsonrpc": "2.0", "id": req_id, "result": result}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
