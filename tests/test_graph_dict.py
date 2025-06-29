import pytest
import sys
import pathlib
import importlib.util
import glob
import os
import torch

# allow imports from repo root
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from compare_encoders import optimizer_to_data


def canonical_representation(opt):
    data = optimizer_to_data(opt)

    def simplify(attrs):
        result = {}
        for k, v in attrs.items():
            key = str(getattr(k, "name", k))
            result[key] = v.tolist() if hasattr(v, "tolist") else v
        return result

    nodes = sorted(
        (int(t), tuple(sorted((k, str(v)) for k, v in simplify(attrs).items())))
        for t, attrs in zip(data.node_types.tolist(), data.node_attributes)
    )
    edges = set(map(tuple, data.edge_index.numpy().T))
    return nodes, edges


@pytest.mark.parametrize(
    "pt_path", glob.glob(os.path.join("computation_graphs", "optimizers", "*.pt")))
def test_graph_dict_matches_source(pt_path):
    base = os.path.splitext(pt_path)[0]
    py_path = base + ".py"
    spec = importlib.util.spec_from_file_location("mod", py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cls_candidates = [v for v in module.__dict__.values() if isinstance(v, type) and issubclass(v, torch.nn.Module)]
    assert len(cls_candidates) == 1, f"Could not find optimizer class in {py_path}"
    cls = cls_candidates[0]

    scripted_py = torch.jit.script(cls())
    scripted_pt = torch.jit.load(pt_path)

    nodes_py, edges_py = canonical_representation(scripted_py)
    nodes_pt, edges_pt = canonical_representation(scripted_pt)

    assert nodes_py == nodes_pt
    assert edges_py == edges_pt
