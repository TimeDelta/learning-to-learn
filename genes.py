from neat.attributes import BaseAttribute, BoolAttribute, StringAttribute, FloatAttribute
from neat.genes import BaseGene
from typing import Dict
import torch

import random
from warnings import warn

from utility import generate_random_string

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
                    setattr(self, attribute_name, node.i(attribute_name))
                elif attribute_type == 'f':
                    attribute = FloatAttribute(attribute_name)
                    setattr(self, attribute_name, node.f(attribute_name))
                elif attribute_type == 's':
                    attribute = StringAttribute(attribute_name)
                    setattr(self, attribute_name, node.s(attribute_name))
                else:
                    warn(f'WARNING: Unknown attribute type for node [{node}]: {attribute_type}')
                self._gene_attributes.append(attribute)
        else:
            self.node_type = None
        if not hasattr(self, 'value'):
            self.value = None
        BaseGene.__init__(self, node_id)

    def copy(self):
        new_gene = NodeGene(self.id)
        new_gene.node_type = self.node_type
        new_gene._gene_attributes = self._gene_attributes.copy()
        for attribute in self._gene_attributes:
            setattr(new_gene, attribute.name, getattr(self, attribute.name))
        return new_gene

    def mutate(self, config):
        # with some probability, add a new attribute
        if random.random() < config.attribute_add_prob:
            r = random.random()
            if r <= .25:
                attr = BoolAttribute(generate_random_string(5))
            elif r <= .5:
                attr = IntAttribute(generate_random_string(5))
            elif r <= .75:
                attr = FloatAttribute(generate_random_string(5))
            else:
                attr = StringAttribute(generate_random_string(5))
            self.add_attribute(attr, config)

        # with some probability, remove an existing one
        if random.random() < config.attribute_delete_prob:
            to_remove = random.choice(self._gene_attributes).name
            while to_remove == 'node_type':
                to_remove = random.choice(self._gene_attributes).name
            self.remove_attribute(to_remove)

    def add_attribute(self, attr: BaseAttribute, config):
        """Add a new attribute to this gene at runtime."""
        self._gene_attributes.append(attr)
        setattr(self, attr.name, attr.init_value(config))

    def remove_attribute(self, name: str):
        """Remove a dynamically added attribute."""
        self._gene_attributes = [a for a in self._gene_attributes if a.name != name]
        delattr(self, name)

    def distance(self, other, config):
        d = 0.0
        for attribute in self._gene_attributes:
            if hasattr(other, attribute.name):
                if getattr(self, attribute.name) != getattr(other, attribute.name):
                    d += 1
            else:
                d += 1
        return d * config.compatibility_weight_coefficient

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if not hasattr(gene2, a.name) or random.random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            elif hasattr(gene2, a.name):
                setattr(new_gene, a.name, getattr(gene2, a.name))
            else:
                warn('Missing Attribute: ' + a.name)
        return new_gene

    def __str__(self):
        return f"NodeGene(id={self.id}, kind={self.kind}, attributes={self.attributes})"

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
        if self.key[1] != other.key[0]:
            d += 1
        if self.enabled != other.enabled:
            d += 1
        return d * config.compatibility_weight_coefficient
