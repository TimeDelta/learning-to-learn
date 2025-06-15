import random

import numpy as np
import torch
from torch_geometric.data import Data

from attributes import FloatAttribute, IntAttribute, StringAttribute
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
from tasks import TASK_FEATURE_DIMS, TASK_TYPE_TO_INDEX
from utility import generate_random_string


def generate_random_dag(num_nodes, num_node_types, edge_prob=0.3):
    types = torch.randint(0, num_node_types, (num_nodes,), dtype=torch.long)
    edges = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if random.random() < edge_prob:
                edges.append([i, j])
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.empty((2, 0), dtype=torch.long)
    )
    dyn_attrs = []
    for _ in range(num_nodes):
        attributes = {}
        for _ in range(random.randint(0, 3)):
            if random.random() <= 0.5:
                attributes[IntAttribute(random.choice(attr_name_vocab))] = random.randint(0, 10)
            if random.random() <= 0.5:
                attributes[FloatAttribute(random.choice(attr_name_vocab))] = random.random() * 5.0
            if random.random() <= 0.5:
                attributes[StringAttribute(random.choice(attr_name_vocab))] = random.choice(attr_name_vocab)
        dyn_attrs.append(attributes)
    return Data(node_types=types, edge_index=edge_index, node_attributes=dyn_attrs)


def generate_data(num_samples, num_node_types):
    graphs, fitnesses = [], []
    task_type = random.choice(list(TASK_FEATURE_DIMS.keys()))
    for _ in range(num_samples):
        graph = generate_random_dag(random.randint(3, 6), num_node_types, edge_prob=0.4)
        graphs.append(graph)
        fitnesses.append({Metric(str(i), "min"): graph.edge_index.size(1) for i in range(fitness_dim)})
    task_features = torch.randn((TASK_FEATURE_DIMS[task_type],))
    return graphs, fitnesses, task_type, task_features


class Metric:
    def __init__(self, name, objective):
        self.name = name
        self.objective = objective


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
    num_node_types = 3
    graph_latent_dim = 16
    task_latent_dim = 8
    fitness_dim = 1

    # lists for both metrics
    res_attention_final = []
    res_attention_auc = []
    res_async_final = []
    res_async_auc = []

    for i in range(100):
        random_seed = random.randint(0, 99999999)
        attr_name_vocab = [generate_random_string(5) for _ in range(20)]
        shared_attr_vocab = SharedAttributeVocab(attr_name_vocab, 5)

        data = generate_data(10, num_node_types)

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
