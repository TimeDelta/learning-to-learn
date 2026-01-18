from typing import Dict, List, Tuple

import torch
import torch._C
import torch.nn as nn
from torch._C import Graph, Node

from genes import NODE_TYPE_OPTIONS, NodeGene
from genome import OptimizerGenome


def build_forward_graph(
    num_nodes: int, edges: List[Tuple[int, int]], input_keys: List[int], output_keys: List[int]
) -> Graph:
    """
    Build a Graph with signature:
      (loss: torch.Tensor, prev_loss: torch.Tensor, named_parameters: List[Tuple[str, Parameter]]) -> Dict[str, Tensor]
    """
    graph = Graph()
    self_val = graph.addInput()

    # add the three inputs
    loss_in = graph.addInput()
    loss_in.setType(torch._C.TensorType.get())
    prev_loss_in = graph.addInput()
    prev_loss_in.setType(torch._C.TensorType.get())
    named_params = graph.addInput()

    # unpack the List[Tuple[str, Tensor]] into separate outputs so treat them as individual tensors in the graph
    unpack = graph.create("prim::ListUnpack", [named_params], len(input_keys) - 2)  # all except loss & prev_loss
    graph.appendNode(unpack)
    param_values = list(unpack.outputs())  # these are Tensors

    # Build a single big Tensor of shape [num_nodes, ...] from [ loss, prev_loss, *param_values ]
    all_inputs = [loss_in, prev_loss_in] + param_values
    lc = graph.create("prim::ListConstruct", all_inputs, 1)
    graph.appendNode(lc)
    stack_node = graph.create("aten::stack", [lc.output()], 1)
    stack_node.i_("dim", 0)
    graph.appendNode(stack_node)
    features = stack_node.output()

    zeros_per_node: List[torch._C.Value] = []
    for _ in range(num_nodes):
        z = graph.create("aten::zeros_like", [loss_in], 1)
        graph.appendNode(z)
        zeros_per_node.append(z.output())
    zeros_list = graph.create("prim::ListConstruct", zeros_per_node, 1)
    graph.appendNode(zeros_list)

    out_stack_init = graph.create("aten::stack", [zeros_list.output()], 1)
    out_stack_init.i_("dim", 0)
    graph.appendNode(out_stack_init)
    out_stack = out_stack_init.output()  # shape [num_nodes, ...]

    # fetch module-held edges and weights
    get_edges = graph.create("prim::GetAttr", [self_val], 1)
    get_edges.s_("name", "edges")
    graph.appendNode(get_edges)
    edges_list = get_edges.output()  # List[Tuple[int, int]]

    get_weights = graph.create("prim::GetAttr", [self_val], 1)
    get_weights.s_("name", "weights")
    graph.appendNode(get_weights)
    weights_list = get_weights.output()  # ParameterList -> behaves like list[Tensor] in TS

    # len(weights) -> trip count
    weights_len = graph.create("aten::len", [weights_list], 1)
    graph.appendNode(weights_len)

    # Loop inputs: (trip_count:int, init_cond:bool, carried: out_stack, edges_list, weights_list, features_stack)
    init_cond = graph.insertConstant(True)
    loop_node = graph.create(
        "prim::Loop", [weights_len.output(), init_cond, out_stack, edges_list, weights_list, features_stack], 4
    )
    graph.appendNode(loop_node)

    # Build loop body
    loop_block = loop_node.addBlock()
    iter_index_val = loop_block.addInput()  # %i : int
    out_stack_in = loop_block.addInput()  # carried 0
    edges_in = loop_block.addInput()  # carried 1
    weights_in = loop_block.addInput()  # carried 2
    features_in = loop_block.addInput()  # carried 3

    # edges[i] -> (src, dst)
    edge_tuple = graph.create("aten::__getitem__", [edges_in, iter_index_val], 1)
    loop_block.appendNode(edge_tuple)
    src_dst_unpack = graph.create("prim::TupleUnpack", [edge_tuple.output()], 2)
    loop_block.appendNode(src_dst_unpack)
    src_idx_dynamic, dst_idx_dynamic = src_dst_unpack.outputs()

    # weights[i] -> Tensor w
    weight_tensor = graph.create("aten::__getitem__", [weights_in, iter_index_val], 1)
    loop_block.appendNode(weight_tensor)

    # Select features[src]  (keep dim as an *input* Value so src can be dynamic)
    dim0_const = graph.insertConstant(0)  # can be captured by block
    select_feat = graph.create("aten::select", [features_in, dim0_const, src_idx_dynamic], 1)
    loop_block.appendNode(select_feat)
    feat_src_val = select_feat.output()

    # mul = feat_src * w
    mul_node = graph.create("aten::mul", [feat_src_val, weight_tensor.output()], 1)
    loop_block.appendNode(mul_node)

    # --- In-place accumulate: out_stack[dst] += mul ---
    # 1) get a view of the destination row
    out_row = graph.create("aten::select", [out_stack_in, dim0_const, dst_idx_dynamic], 1)
    loop_block.appendNode(out_row)
    # 2) add_ in place (alpha=1.0)
    add_inplace = graph.create("aten::add_", [out_row.output(), mul_node.output()], 1)
    add_inplace.f_("alpha", 1.0)
    loop_block.appendNode(add_inplace)
    # Note: the in-place op mutates out_stack_in via the view; we return out_stack_in as carried.

    # loop continuation condition (ignored since trip-count drives, but must be True)
    cond_next = graph.insertConstant(True)

    # Register block outputs in order: (cond_next, carried...)
    loop_block.registerOutput(cond_next)
    loop_block.registerOutput(out_stack_in)  # mutated by add_
    loop_block.registerOutput(edges_in)
    loop_block.registerOutput(weights_in)
    loop_block.registerOutput(features_in)

    # Loop outputs correspond 1:1 to carried vars
    out_stack_final = loop_node.output(0)
    # loop_node.output(1..3) are edges/weights/features (unchanged); we ignore them.

    # === Build Dict[str, Tensor] for requested outputs ===
    dict_items: List[torch._C.Value] = []
    for ok in output_keys:
        key_str = graph.insertConstant(str(ok))
        dict_items.append(key_str)

        if ok == 0:
            dict_items.append(loss_in)
        elif ok == 1:
            dict_items.append(prev_loss_in)
        elif 2 <= ok < 2 + len(param_values):
            dict_items.append(param_values[ok - 2])
        else:
            # take computed feature for this node id
            idx_const = graph.insertConstant(int(ok))
            pick_out = graph.create("aten::select", [out_stack_final, dim0_const, idx_const], 1)
            graph.appendNode(pick_out)
            dict_items.append(pick_out.output())

    dlc = graph.create("prim::ListConstruct", dict_items, 1)
    graph.appendNode(dlc)
    dict_node = graph.create("prim::DictConstruct", [dlc.output()], 1)
    graph.appendNode(dict_node)
    graph.registerOutput(dict_node.output())

    return graph


class DynamicOptimizerModule(nn.Module):
    """Simple PyTorch module implementing one round of message passing."""

    def __init__(self, genome, input_keys, output_keys, graph_dict=None):
        super().__init__()
        self.num_nodes = len(genome.nodes)
        edges: List[Tuple[int, int]] = []
        jit_weights: List[torch.Tensor] = []
        self.weights = nn.ParameterList()
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                edges.append((src, dst))
                w = getattr(conn, "weight", 1.0)
                param = nn.Parameter(torch.tensor(w))
                self.weights.append(param)
                jit_weights.append(param)

        self.edges = torch.jit.Attribute(edges, List[Tuple[int, int]])
        self._jit_weights = torch.jit.Attribute(jit_weights, List[torch.Tensor])
        if graph_dict is not None and "node_types" in graph_dict:
            node_types_val = graph_dict["node_types"]
            if isinstance(node_types_val, torch.Tensor):
                node_types_tensor = node_types_val.clone().detach()
            else:
                node_types_tensor = torch.as_tensor(node_types_val)
        else:
            node_types_tensor = torch.empty(0, dtype=torch.long)
        self.node_types = torch.jit.Attribute(node_types_tensor, torch.Tensor)
        self.input_keys = torch.jit.Attribute(list(input_keys), List[int])
        self.output_keys = torch.jit.Attribute(list(output_keys), List[int])

    def forward(
        self,
        loss: torch.Tensor,
        prev_loss: torch.Tensor,
        named_parameters: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        base_features = torch.jit.annotate(List[torch.Tensor], [])
        base_features.append(loss)
        base_features.append(prev_loss)
        for _, param in named_parameters:
            base_features.append(param)

        edge_list = self.edges
        max_index = 0
        for src, dst in edge_list:
            if src > max_index:
                max_index = src
            if dst > max_index:
                max_index = dst

        num_slots = max(self.num_nodes, len(base_features), max_index + 1)

        computed = torch.jit.annotate(List[torch.Tensor], [])
        for idx in range(num_slots):
            if idx < len(base_features):
                computed.append(torch.zeros_like(base_features[idx]))
            else:
                computed.append(torch.zeros_like(loss))

        weight_list = self._jit_weights
        num_base = len(base_features)

        for idx in range(len(weight_list)):
            src, dst = edge_list[idx]
            if dst >= len(computed) or dst < num_base:
                continue
            if src < num_base:
                src_val = base_features[src]
            elif src < len(computed):
                src_val = computed[src]
            else:
                src_val = loss
            if computed[dst].shape != src_val.shape:
                computed[dst] = torch.zeros_like(src_val)
            computed[dst] = computed[dst] + src_val * weight_list[idx]

        all_values = torch.jit.annotate(List[torch.Tensor], [])
        for idx in range(num_slots):
            if idx < num_base:
                all_values.append(base_features[idx])
            else:
                all_values.append(computed[idx])

        outputs = torch.jit.annotate(Dict[str, torch.Tensor], {})
        for ok in self.output_keys:
            idx = int(ok)
            if 0 <= idx < len(all_values):
                outputs[str(idx)] = all_values[idx]
            else:
                outputs[str(idx)] = torch.zeros_like(loss)
        return outputs


def genome_from_graph_dict(graph_dict, genome_config, key=None) -> OptimizerGenome:
    """Instantiate an OptimizerGenome's nodes/connections from a decoded graph dict."""
    genome = OptimizerGenome(key)
    genome.nodes = {}
    node_types_val = graph_dict.get("node_types")
    if node_types_val is None:
        raise ValueError("graph_dict missing node_types")
    if isinstance(node_types_val, torch.Tensor):
        node_type_indices = node_types_val.clone().detach().view(-1).tolist()
    else:
        node_type_indices = list(node_types_val)
    node_attrs_seq = graph_dict.get("node_attributes", [])
    for nid, type_idx in enumerate(node_type_indices):
        ng = NodeGene(nid, None)
        attr_dict = node_attrs_seq[nid] if nid < len(node_attrs_seq) else {}
        node_type_name = attr_dict.get("node_type")
        if node_type_name is None:
            try:
                node_type_name = NODE_TYPE_OPTIONS[int(type_idx)]
            except (ValueError, TypeError, IndexError):
                node_type_name = "hidden"
        ng.node_type = node_type_name
        ng.dynamic_attributes = dict(attr_dict)
        genome.nodes[nid] = ng
    genome.next_node_id = len(genome.nodes)

    genome.connections = {}
    edge_index_val = graph_dict.get("edge_index")
    if edge_index_val is not None:
        if isinstance(edge_index_val, torch.Tensor):
            edge_tensor = edge_index_val.clone().detach().long()
        else:
            edge_tensor = torch.as_tensor(edge_index_val, dtype=torch.long)
        if edge_tensor.dim() == 1:
            edge_tensor = edge_tensor.view(2, -1)
        if edge_tensor.numel() > 0:
            for src, dst in edge_tensor.t().tolist():
                cg = genome.create_connection(genome_config, src, dst)
                cg.enabled = True
                genome.connections[(src, dst)] = cg

    return genome


def rebuild_and_script(graph_dict, config, key) -> DynamicOptimizerModule:
    """
    1) Rebuild the genome nodes+connections (as before)
    2) Create ScriptModule, attach `w_src_dst` Parameters
    3) Generate the IR with `build_forward_graph` and hook it up
    """
    genome = genome_from_graph_dict(graph_dict, config, key)

    # --- build a Python module and script it ---
    module = DynamicOptimizerModule(genome, config.input_keys, config.output_keys, graph_dict)
    return torch.jit.script(module)
