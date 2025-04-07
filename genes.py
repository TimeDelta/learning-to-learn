from neat.attributes import BoolAttribute, StringAttribute
from neat.genes import BaseGene

# modified from neat-python versions

class NodeGene(BaseGene):
    _gene_attributes = [
        StringAttribute('activation', options=''),
        StringAttribute('aggregation', options=''),
    ]

    def __init__(self, node_id: int, activation, aggregation):
        assert isinstance(node_id, int), f"NodeGene id must be an int, not {node_id!r}"
        BaseGene.__init__(self, node_id)
        self.node_type = 'normal' # ['input', 'normal', 'output']
        self.activation = activation
        self.aggregation = aggregation

    def copy(self):
        return NodeGene(self.id, self.node_type, self.activation, self.aggregation)

    def distance(self, other, config):
        d = 0.0
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient

    def __str__(self):
        return f"NodeGene(id={self.id}, type={self.node_type}, activation={self.activation}, aggregation={self.aggregation})"

class ConnectionGene(BaseGene):
    _gene_attributes = [
        BoolAttribute('enabled'),
    ]

    def __init__(self, key):
        assert isinstance(key, tuple), f"ConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = 0.0
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
