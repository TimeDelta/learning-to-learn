import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from population_visualizer import (
    DECODED_GRAPH_DICT_KEY,
    RenderContext,
    build_mermaid_graph,
    save_summary,
    select_snapshot_entries,
)


def test_select_snapshot_entries_filters_and_sorts():
    snapshot = {
        "entries": [
            {"genome_id": 2, "fitness": 0.3},
            {"genome_id": 1, "fitness": 0.9},
            {"genome_id": 3, "fitness": 0.5, "invalid_graph": True},
        ]
    }
    ranked = select_snapshot_entries(snapshot, top_k=2)
    assert [entry["genome_id"] for entry in ranked] == [1, 2]

    ranked_all = select_snapshot_entries(snapshot, include_invalid=True, sort_by="genome_id")
    assert [entry["genome_id"] for entry in ranked_all] == [1, 2, 3]

    filtered = select_snapshot_entries(snapshot, genome_ids=[2])
    assert [entry["genome_id"] for entry in filtered] == [2]


def test_build_mermaid_graph_outputs_basic_structure():
    entry = {
        "genome_id": 7,
        "species_id": 3,
        "fitness": 1.5,
        "graph": {
            "node_types": [0, 1, 2],
            "edge_index": [(0, 1), (1, 2)],
            "node_attributes": [
                {"alpha": 0.1},
                {"beta": [1, 2]},
                {"gamma": {"x": 1}},
            ],
        },
    }
    mermaid = build_mermaid_graph(entry, context=RenderContext(generation=2, rank=1, task="demo"), max_attr_lines=2)
    assert "graph LR" in mermaid
    assert "gen=2" in mermaid
    assert "node_0 --> node_1" in mermaid
    assert "node_1 --> node_2" in mermaid


def test_build_mermaid_graph_renders_line_breaks_without_ellipsis():
    entry = {
        "genome_id": 5,
        "graph": {
            "node_types": [0],
            "edge_index": [],
            "node_attributes": [
                {
                    "pin_role": "hidden",
                    "pin_slot_index": 0,
                    "extra": "value",
                }
            ],
        },
    }
    mermaid = build_mermaid_graph(entry)
    assert "<br/>pin_role=hidden" in mermaid
    assert "<br/>pin_slot_index=0" in mermaid
    assert "…" not in mermaid


def test_build_mermaid_graph_ignores_predicted_edges_when_missing():
    entry = {
        "genome_id": 9,
        "graph": {
            "node_types": [0, 1],
            "edge_index": [],
            "node_attributes": [{}, {}],
        },
    }
    entry["graph"][DECODED_GRAPH_DICT_KEY] = {
        "node_types": [0, 1],
        "edge_index": [(0, 1)],
        "node_attributes": [{}, {}],
    }
    mermaid = build_mermaid_graph(entry)
    assert "node_0 --> node_1" not in mermaid
    assert "Graph has no edges" in mermaid


def test_save_summary_serializes_entries(tmp_path):
    path = tmp_path / "summary.json"
    entries = [{"genome_id": 4, "species_id": 0, "fitness": 0.25, "invalid_graph": False, "invalid_reason": None}]
    save_summary(entries, path, extra={"generation": 5, "task": "demo"})
    payload = json.loads(path.read_text())
    assert payload["generation"] == 5
    assert payload["entries"][0]["genome_id"] == 4
