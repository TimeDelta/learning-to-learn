import torch
import torch.jit


@torch.jit.script
class Node:
    def __init__(self, gene):
        self.id = gene.id
        self.node_type = gene.node_type
        self.activate = gene.activation
        self.aggregate = gene.aggregation

        self.inputs: List[float] = []
        self.output: float = 0.0

    def __call__(self):
        self.output = self.inputs
        if self.aggregation:
            self.output = self.aggregate(self.output)
        if self.activation:
            self.output = self.activate(self.output)
        return self.output
