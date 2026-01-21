from pathlib import Path

import torch

debug_dir = Path("debug_guided_offspring")
for path in sorted(debug_dir.glob("*.pt")):
    graph = torch.load(path)
    edge_index = graph.get("edge_index")
    node_types = graph.get("node_types")
    attrs = graph.get("node_attributes")

    if edge_index is not None:
        if edge_index.dim() == 1:
            edges = edge_index.numel() // 2
        else:
            edges = edge_index.size(1)
    else:
        edges = 0

    print(f"{path.name}: edges={edges}, nodes={0 if node_types is None else node_types.numel()}")
    # Uncomment for a deeper peek at the first few edges/attributes
    if edge_index is not None and edge_index.numel():
        print("  sample edges:", edge_index[:, : min(5, edge_index.size(1))].tolist())
    if attrs:
        print("  attr[0]:", attrs[0] if attrs else None)
