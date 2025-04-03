from neat.genes import BaseGene

class NodeGene(BaseGene):
    def __init__(self,
        node_id: int,
        node_type: str,
        activation: ActivationFunction,
        aggregation: AggregationFunction
    ):
        self.id = node_id
        self.node_type = node_type # "input", "hidden", "output"
        self.activation = activation
        self.aggregation = aggregation

    def copy(self):
        return NodeGene(self.id, self.node_type, self.activation, self.aggregation)

    def mutate_activation(self, new_activation: ActivationFunction):
        self.activation = new_activation

    def mutate_aggregation(self, new_aggregation: AggregationFunction):
        self.aggregation = new_aggregation

    def activate(self, x: float) -> float:
        return self.activation.activate(x)

    def aggregate(self, inputs: List[float]) -> float:
        return self.aggregation.aggregate(inputs)

    def __str__(self):
        return f"NodeGene(id={self.id}, type={self.node_type}, activation={self.activation}, aggregation={self.aggregation})"
