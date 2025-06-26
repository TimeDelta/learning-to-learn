import os
import random
import time
import tracemalloc

import numpy as np
import torch
from torch_geometric.data import Data

from attributes import BoolAttribute, FloatAttribute, IntAttribute, StringAttribute
from search_space_compression import (
    AsyncGraphEncoder,
    FitnessPredictor,
    GraphDecoder,
    GraphEncoder,
    NodeAttributeDeepSetEncoder,
    OnlineTrainer,
    SelfCompressingFitnessRegularizedDAGVAE,
    SharedAttributeVocab,
    TasksEncoder,
)
from tasks import TASK_TYPE_TO_CLASS
from models import ManyLossMinimaModel
from metrics import MSELoss
from genes import NODE_TYPE_TO_INDEX


# default until dataset generation sets real number of fitness dimensions
fitness_dim = 1

optimizer_paths = [
    os.path.join("computation_graphs/optimizers", f)
    for f in os.listdir("computation_graphs/optimizers")
    if f.endswith(".pt")
]


def optimizer_to_data(opt):
    node_map = {}
    node_types = []
    node_attrs = []
    for idx, node in enumerate(opt.graph.nodes()):
        node_map[node] = idx
        node_types.append(NODE_TYPE_TO_INDEX.get(node.kind(), NODE_TYPE_TO_INDEX["hidden"]))
        attrs = {}
        for name in node.attributeNames():
            kind = node.kindOf(name)
            if kind == "i":
                attrs[IntAttribute(name)] = node.i(name)
            elif kind == "f":
                attrs[FloatAttribute(name)] = node.f(name)
            elif kind == "s":
                attrs[StringAttribute(name)] = node.s(name)
        node_attrs.append(attrs)

    edges = []
    for node in opt.graph.nodes():
        dst = node_map[node]
        for inp in node.inputs():
            src = inp.node()
            if src in node_map:
                edges.append([node_map[src], dst])
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    node_types = torch.tensor(node_types, dtype=torch.long)
    return Data(node_types=node_types, edge_index=edge_index, node_attributes=node_attrs)


def evaluate_optimizer(optimizer, model, task, steps=5):
    tracemalloc.start()
    start = time.perf_counter()
    prev_metrics_values = torch.tensor([0.0] * len(task.metrics))
    for _ in range(steps):
        metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
        new_params = optimizer(metrics_values, prev_metrics_values, model.named_parameters())
        model.load_state_dict(new_params)
        prev_metrics_values = task.evaluate_metrics(model, task.train_data).requires_grad_()
    stop = time.perf_counter()
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    time_cost = stop - start
    validation_metrics = task.evaluate_metrics(model, task.valid_data).detach().data.numpy()
    validation_metrics = {m: float(validation_metrics[i]) for i, m in enumerate(task.metrics)}
    validation_metrics[MSELoss()] = validation_metrics.get(MSELoss(), 0.0)
    return validation_metrics, time_cost, peak_memory


def generate_data(num_samples):
    graphs, fitnesses = [], []
    attr_names = set()
    task_type = random.choice(list(TASK_TYPE_TO_CLASS.keys()))
    task = TASK_TYPE_TO_CLASS[task_type].random_init(num_samples=50, silent=True)
    for i in range(num_samples):
        opt = torch.jit.load(optimizer_paths[i % len(optimizer_paths)])
        graph = optimizer_to_data(opt)
        graphs.append(graph)
        for attrs in graph.node_attributes:
            for attr in attrs:
                attr_names.add(attr.name)
        fitness_dict, _, _ = evaluate_optimizer(opt, ManyLossMinimaModel(task.train_data.num_input_features), task)
        fitnesses.append(fitness_dict)
    task_features = torch.tensor(np.concatenate(task.features, axis=0), dtype=torch.float32)
    return graphs, fitnesses, task_type, task_features, sorted(attr_names)


def train_model(encoder_cls, train_graphs, random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    attr_encoder = NodeAttributeDeepSetEncoder(shared_attr_vocab, 10, 20, 50)
    graph_encoder = encoder_cls(num_node_types, attr_encoder, graph_latent_dim, hidden_dims=[16])
    task_encoder = TasksEncoder(hidden_dim=16, latent_dim=task_latent_dim, type_embedding_dim=8)
    decoder = GraphDecoder(num_node_types, graph_latent_dim, shared_attr_vocab)
    predictor = FitnessPredictor(latent_dim=graph_latent_dim + task_latent_dim, hidden_dim=32, fitness_dim=fitness_dim)
    model = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, task_encoder, decoder, predictor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer = OnlineTrainer(model, optimizer)
    trainer.add_data(*train_graphs)
    history = trainer.train(epochs=10, batch_size=4, warmup_epochs=1, verbose=False)

    # compute per-epoch summed losses
    loss_seq = [h.sum().item() for h in history]
    final_loss = loss_seq[-1]
    # compute area under curve (AUC) over epochs
    auc = np.trapz(loss_seq, dx=1)
    return final_loss, auc


if __name__ == "__main__":
    num_node_types = len(NODE_TYPE_TO_INDEX)
    graph_latent_dim = 16
    task_latent_dim = 8

    # lists for both metrics
    res_attention_final = []
    res_attention_auc = []
    res_async_final = []
    res_async_auc = []

    for i in range(100):
        random_seed = random.randint(0, 99999999)
        data = generate_data(10)
        attr_name_vocab = data[4]
        shared_attr_vocab = SharedAttributeVocab(attr_name_vocab, 5)
        fitness_dim = len(data[1][0])

        final_att, auc_att = train_model(GraphEncoder, data, random_seed)
        final_async, auc_async = train_model(AsyncGraphEncoder, data, random_seed)

        res_attention_final.append(final_att)
        res_attention_auc.append(auc_att)
        res_async_final.append(final_async)
        res_async_auc.append(auc_async)

        print(f"  attention encoder → final loss: {final_att:.4f}, AUC loss: {auc_att:.4f}")
        print(f"  async encoder     → final loss: {final_async:.4f}, AUC loss: {auc_async:.4f}")

    print("\nSummary:")
    print(f"Mean final loss (attention encoder): {np.mean(res_attention_final):.4f}")
    print(f"Mean AUC loss   (attention encoder): {np.mean(res_attention_auc):.4f}")
    print(f"Mean final loss (async encoder):    {np.mean(res_async_final):.4f}")
    print(f"Mean AUC loss   (async encoder):    {np.mean(res_async_auc):.4f}")
