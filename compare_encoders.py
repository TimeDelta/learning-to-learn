import argparse
import os
import random
import time
import tracemalloc

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import mlflow
from attributes import BoolAttribute, FloatAttribute, IntAttribute, StringAttribute
from genes import NODE_TYPE_TO_INDEX
from metrics import MSELoss, sort_metrics_by_name
from models import ManyLossMinimaModel
from search_space_compression import (
    AsyncGraphEncoder,
    FitnessPredictor,
    GraphDecoder,
    GraphEncoder,
    NodeAttributeDeepSetEncoder,
    OnlineTrainer,
    SelfCompressingFitnessRegularizedDAGVAE,
    SharedAttributeVocab,
)
from tasks import TASK_TYPE_TO_CLASS

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


def evaluate_fitness_loss(model, graphs, fitnesses, task_type, task_features, batch_size=4):
    """Return average MSE loss over a dataset."""
    dataset = []
    for graph, fitness_dict in zip(graphs, fitnesses):
        data = graph.clone()
        fitness = [f[1] for f in sorted(fitness_dict.items(), key=lambda item: item[0].name)]
        data.y = torch.tensor(fitness, dtype=torch.float)
        dataset.append(data)

    loader = DataLoader(dataset, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, pred, *_ = model(
                batch.node_types,
                batch.edge_index,
                batch.node_attributes,
                batch.batch,
                teacher_attr_targets=None,
                num_graphs=batch.num_graphs,
            )
            loss = torch.nn.functional.mse_loss(pred, batch.y.to(device))
            losses.append(loss.item())
    return float(np.mean(losses))


def train_model(encoder_cls, full_dataset, random_seed, val_ratio=0.2):
    graphs, fitnesses, task_type, task_features = full_dataset
    combined = list(zip(graphs, fitnesses))
    random.seed(random_seed)
    random.shuffle(combined)
    val_size = max(1, int(len(combined) * val_ratio))
    val_pairs = combined[:val_size]
    train_pairs = combined[val_size:]
    train_graphs, train_fitnesses = zip(*train_pairs)
    val_graphs, val_fitnesses = zip(*val_pairs)

    random.seed(random_seed)
    torch.manual_seed(random_seed)

    attr_encoder = NodeAttributeDeepSetEncoder(shared_attr_vocab, 10, 20, 20)
    graph_encoder = encoder_cls(num_node_types, attr_encoder, graph_latent_dim, hidden_dims=[16])
    decoder = GraphDecoder(num_node_types, graph_latent_dim, shared_attr_vocab)
    predictor = FitnessPredictor(
        latent_dim=graph_latent_dim,
        hidden_dim=32,
        fitness_dim=fitness_dim,
        icnn_hidden_dims=(graph_latent_dim, graph_latent_dim // 2 or 1),
    )
    model = SelfCompressingFitnessRegularizedDAGVAE(graph_encoder, decoder, predictor)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    metric_keys = sort_metrics_by_name(train_fitnesses[0].keys()) if train_fitnesses else []
    trainer = OnlineTrainer(model, optimizer, metric_keys=metric_keys)
    trainer.add_data(train_graphs, train_fitnesses)

    loss_history = trainer.train(epochs=10, batch_size=4, warmup_epochs=10, verbose=True)
    train_losses = [lh.sum().item() for lh in loss_history]
    val_loss = evaluate_fitness_loss(model, val_graphs, val_fitnesses, task_type, task_features)

    final_loss = train_losses[-1]
    auc = np.trapz(train_losses, dx=1)
    return final_loss, auc, val_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare graph encoders")
    parser.add_argument("--num-runs", type=int, default=1, help="number of runs")
    parser.add_argument("--num-samples", type=int, default=1000, help="number of optimizers to sample")
    parser.add_argument("--experiment-name", type=str, default="compare_encoders", help="MLflow experiment name")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    num_node_types = len(NODE_TYPE_TO_INDEX)
    graph_latent_dim = 16

    res_attention_final = []
    res_attention_auc = []
    res_attention_val = []
    res_attention_val_auc = []
    res_async_final = []
    res_async_auc = []
    res_async_val = []
    res_async_val_auc = []

    for i in range(args.num_runs):
        random_seed = random.randint(0, 99999999)
        data = generate_data(args.num_samples)
        attr_name_vocab = data[4]
        globals()["shared_attr_vocab"] = SharedAttributeVocab(attr_name_vocab, 50)
        globals()["fitness_dim"] = len(data[1][0])

        final_att, auc_att, val_att = train_model(GraphEncoder, data[:4], random_seed)
        with mlflow.start_run(run_name=f"GraphEncoder_{i}"):
            mlflow.log_params({"encoder": "GraphEncoder", "seed": random_seed, "num_samples": args.num_samples})
            mlflow.log_metrics(
                {
                    "train_final_loss": final_att,
                    "train_auc": auc_att,
                    "val_loss": val_att,
                }
            )

        final_async, auc_async, val_async = train_model(AsyncGraphEncoder, data[:4], random_seed)
        with mlflow.start_run(run_name=f"AsyncGraphEncoder_{i}"):
            mlflow.log_params({"encoder": "AsyncGraphEncoder", "seed": random_seed, "num_samples": args.num_samples})
            mlflow.log_metrics(
                {
                    "train_final_loss": final_async,
                    "train_auc": auc_async,
                    "val_loss": val_async,
                }
            )

        res_attention_final.append(final_att)
        res_attention_auc.append(auc_att)
        res_attention_val.append(val_att)
        res_async_final.append(final_async)
        res_async_auc.append(auc_async)
        res_async_val.append(val_async)

        print(f"  attention encoder → train loss: {final_att:.4f}, val loss: {val_att:.4f}, AUC loss: {auc_att:.4f}")
        print(
            f"  async encoder     → train loss: {final_async:.4f}, val loss: {val_async:.4f}, AUC loss: {auc_async:.4f}"
        )

    print("\nSummary:")
    print(f"Mean train loss (attention encoder): {np.mean(res_attention_final):.4f}")
    print(f"Mean val loss   (attention encoder): {np.mean(res_attention_val):.4f}")
    print(f"Mean AUC loss   (attention encoder): {np.mean(res_attention_auc):.4f}")
    print(f"Mean val AUC    (attention encoder): {np.mean(res_attention_val_auc):.4f}")
    print(f"Mean train loss (async encoder):    {np.mean(res_async_final):.4f}")
    print(f"Mean val loss   (async encoder):    {np.mean(res_async_val):.4f}")
    print(f"Mean AUC loss   (async encoder):    {np.mean(res_async_auc):.4f}")
    print(f"Mean val AUC    (async encoder):    {np.mean(res_async_val_auc):.4f}")
