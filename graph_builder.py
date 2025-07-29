from typing import Dict, List, Tuple

import torch
import torch._C
import torch.nn as nn
from torch._C import Graph, Node

from genes import NodeGene
from genome import OptimizerGenome


def build_forward_graph(
    num_nodes: int, edges: List[Tuple[int, int]], input_keys: List[int], output_keys: List[int]
) -> Graph:
    """
    Build a Graph with signature:
      (Tensor, Tensor, List[Tuple[str, Tensor]]) -> Dict[str, Tensor]
    """
    graph = Graph()
    self_val = graph.addInput()

    # 1) add the three inputs
    loss_in = graph.addInput()
    loss_in.setType(torch._C.TensorType.get())
    prev_loss_in = graph.addInput()
    prev_loss_in.setType(torch._C.TensorType.get())
    named_params = graph.addInput()

    # 2) unpack the List[Tuple[str, Tensor]] into separate outputs
    #    so we can treat them as individual tensors in the graph
    unpack = graph.create("prim::ListUnpack", [named_params], len(input_keys) - 2)  # all except loss & prev_loss
    graph.appendNode(unpack)
    param_values = list(unpack.outputs())  # these are Tensors

    # 3) Build a single big Tensor of shape [num_nodes, ...] from
    #    [ loss, prev_loss, *param_values ]
    all_inputs = [loss_in, prev_loss_in] + param_values
    lc = graph.create("prim::ListConstruct", all_inputs, 1)
    graph.appendNode(lc)
    zero = graph.insertConstant(0)
    st = graph.create("aten::stack", [lc.output(), zero], 1)
    graph.appendNode(st)
    features = st.output()

    # 4) do one round of message‐passing: out_feats[n] = sum_in_edges w_e * features[src]
    #    To keep the example small, we use an unrolled sequence of element‐wise ops.
    #    A production version would vectorize with scatter_add as shown earlier.
    #    Here we just demonstrate Node usage.
    accum = graph.insertConstant(None)  # placeholder; we’ll replace per-node below

    # For each node id in sorted order, build:
    #    tmp = 0
    #    for (src, dst) in edges if dst == nid:
    #        w = self.w_src_dst        # captured later as a submodule parameter
    #        mul = features[src] * w
    #        tmp = tmp + mul
    #    out_feats[nid] = tmp
    node_outputs: Dict[int, Node] = {}
    for nid in range(num_nodes):
        # start `tmp = 0`
        zero_t = graph.create("aten::zeros_like", [loss_in], 1)
        graph.appendNode(zero_t)
        tmp = zero_t.output()

        for src, dst in edges:
            if dst != nid:
                continue
            # load the weight parameter by name: w_{src}_{dst} in the Module
            w_node = graph.create("prim::GetAttr", [self_val], 1)
            w_node.s_("name", f"w_{src}_{dst}")
            graph.appendNode(w_node)
            w_out = w_node.output()

            # gather features[src]
            src_idx = graph.insertConstant(src)
            pick = graph.create("aten::select", [features, zero, src_idx], 1)
            graph.appendNode(pick)
            feat_src = pick.output()

            # mul = feat_src * w_out
            mul = graph.create("aten::mul", [feat_src, w_out], 1)
            graph.appendNode(mul)

            # tmp = tmp + mul
            add = graph.create("aten::add", [tmp, mul.output()], 1)
            graph.appendNode(add)
            tmp = add.output()

        node_outputs[nid] = tmp

    # 4b) Build a single map of ALL node‐IDs → their torch.Value,
    #     treating node 0/1/2+i as the three forwarded inputs.
    all_outputs: Dict[int, torch._C.Value] = dict(node_outputs)
    # node 0 := loss, node 1 := prev_loss
    all_outputs[0] = loss_in
    all_outputs[1] = prev_loss_in
    # nodes 2,3,4… correspond to each named_parameters[i]
    for i, val in enumerate(param_values):
        all_outputs[2 + i] = val

    # 5) register outputs: map each output_key to its computed tensor,
    #    but we must return a Dict[str, Tensor], so we build a DictConstruct.
    #    First build a ListConstruct of [ key1, val1, key2, val2, ... ]
    dict_list_elems = []
    for ok in output_keys:
        # key string
        ks = graph.insertConstant(str(ok))
        dict_list_elems.append(ks)
        dict_list_elems.append(all_outputs[ok])

    dc = graph.create("prim::ListConstruct", dict_list_elems, 1)
    graph.appendNode(dc)
    # finally pack to Dict
    dict_node = graph.create("prim::DictConstruct", [dc.output()], 1)
    graph.appendNode(dict_node)
    graph.registerOutput(dict_node.output())

    return graph


class DynamicOptimizerModule(nn.Module):
    """Simple PyTorch module implementing one round of message passing."""

    def __init__(self, genome, input_keys, output_keys, graph_dict=None):
        super().__init__()
        self.num_nodes = len(genome.nodes)
        self.edges: List[Tuple[int, int]] = []
        self.weights = nn.ParameterList()
        self.node_types: List[str] = []
        for (src, dst), conn in genome.connections.items():
            if conn.enabled:
                self.edges.append((src, dst))
                w = getattr(conn, "weight", 1.0)
                self.weights.append(nn.Parameter(torch.tensor(w)))
        for nid in range(self.num_nodes):
            ng = genome.nodes[nid]
            self.node_types.append(ng.node_type)

        self.input_keys = input_keys
        self.output_keys = output_keys
        self.graph_dict = graph_dict

    def forward(
        self,
        loss: torch.Tensor,
        prev_loss: torch.Tensor,
        named_parameters: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        params = [p for _, p in named_parameters]
        all_inputs = [loss, prev_loss] + params
        features = torch.stack(all_inputs, 0)

        out_feats = [torch.zeros_like(loss) for _ in range(self.num_nodes)]
        for idx, w in enumerate(self.weights):
            src, dst = self.edges[idx]
            out_feats[dst] = out_feats[dst] + features[src] * w

        all_outputs = list(out_feats)
        all_outputs[0] = loss
        all_outputs[1] = prev_loss
        for i, p in enumerate(params):
            if 2 + i < len(all_outputs):
                all_outputs[2 + i] = p

        outputs = {}
        for ok in self.output_keys:
            outputs[str(ok)] = all_outputs[ok]
        return outputs


def rebuild_and_script(graph_dict, config, key) -> DynamicOptimizerModule:
    """
    1) Rebuild the genome nodes+connections (as before)
    2) Create ScriptModule, attach `w_src_dst` Parameters
    3) Generate the IR with `build_forward_graph` and hook it up
    """
    # --- rebuild genome structure ---
    genome = OptimizerGenome(key)

    # nodes
    node_types = graph_dict["node_types"].tolist()
    node_attrs = graph_dict["node_attributes"]
    for nid, _ in enumerate(node_types):
        ng = NodeGene(nid, None)
        ng.node_type = node_attrs[nid].get("node_type")
        ng.dynamic_attributes = dict(node_attrs[nid])
        genome.nodes[nid] = ng
    genome.next_node_id = len(node_types)

    # connections
    for src, dst in graph_dict["edge_index"].t().tolist():
        cg = genome.create_connection(config, src, dst)
        cg.enabled = True
        genome.connections[(src, dst)] = cg

    # --- build a Python module and script it ---
    if genome.connections:
        module = DynamicOptimizerModule(genome, config.input_keys, config.output_keys, graph_dict)
        return torch.jit.script(module)
    return None
