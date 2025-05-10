from neat.attributes import BoolAttribute, StringAttribute, FloatAttribute
from neat.genes import BaseGene
from typing import Dict
import torch

class IntAttribute(FloatAttribute):
    def clamp(self, value, config):
        return int(super.clamp(value, config))

    def init_value(self, config):
        return int(super.init_value(config))

    def mutate_value(self, value, config):
        return int(super.mutate_value(value, config))

# modified from neat-python versions
node_type_options = [ # for mutation
    'aten::add',
    'aten::sub',
    'aten::mul',
    'aten::div',
    'aten::pow',
    'aten::matmul',
    'aten::transpose',
    'aten::max',
    'aten::min',
    'aten::sum',
    'aten::len',
]

class NodeGene(BaseGene):
    _gene_attributes = [
        StringAttribute('node_type', options=','.join(node_type_options))
    ]
    def __init__(self, node_id: int, node: torch._C.Node=None):
        if node:
            self.node_type = node.kind()
            for attribute_name in node.attributeNames():
                attribute_type = node.kindOf(attribute_name)
                if attribute_type == 'i':
                    attribute = IntAttribute(attribute_name)
                elif attribute_type == 'f':
                    attribute = FloatAttribute(attribute_name)
                elif attribute_type == 's':
                    attribute = StringAttribute(attribute_name)
                else:
                    print(f'WARNING: Unknown attribute type for node [{node}]')
                self._gene_attributes.append(attribute)
        BaseGene.__init__(self, node_id)

    def copy(self):
        new_gene = TSNodeGene(self.id, self.node)
        new_gene.kind = self.kind
        new_gene.attributes = self.attributes.copy()
        return new_gene

    def distance(self, other, config):
        d = 0.0
        for attribute in self._gene_attributes:
            if hasattr(other, attribute.name):
                if getattr(self, attribute.name) != getattr(other, attribute.name):
                    d += 1
            else:
                d += 1
        return d * config.compatibility_weight_coefficient

    def __str__(self):
        return f"TSNodeGene(id={self.id}, kind={self.kind}, attributes={self.attributes})"

class ConnectionGene(BaseGene):
    _gene_attributes = [
        BoolAttribute('enabled'),
    ]

    def __init__(self, key):
        assert isinstance(key, tuple), f"ConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)
        self.enabled = True
        self.innovation = 0

    def copy(self):
        new_conn = ConnectionGene(self.key)
        new_conn.enabled = self.enabled
        new_conn.innovation = self.innovation
        return new_conn

    def __str__(self):
        return (f"TSConnectionGene(in_node={self.in_node}, out_node={self.out_node}, enabled={self.enabled}, innovation={self.innovation})")

    def distance(self, other, config):
        d = 0.0
        if self.key[0] != other.key[0]:
            d += 1
        if key[1] != other.key[0]:
            d += 1
        if self.enabled != other.enabled:
            d += 1
        return d * config.compatibility_weight_coefficient
