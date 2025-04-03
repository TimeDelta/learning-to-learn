class Node:
    def __init__(self, gene):
        self.id = gene.id
        self.node_type = gene.node_type
        self.activation = gene.activation
        self.aggregation = gene.aggregation

        self.inputs: List[float] = []
        self.output: float = 0.0

    def __call__(self):
        if self.aggregation:
            self.output = self.aggregate(self.inputs)
        elif len(self.inputs) > 1:
            raise Exception(f'Aggregation function required for {self.node_type} node {self.id}')
        if self.activation:
            self.output = self.gene.activate(self.output)
        return self.output
