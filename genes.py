from neat.genes import BaseGene
from typing import Dict
import torch

import random
from warnings import warn

from attributes import BaseAttribute, BoolAttribute, StringAttribute, FloatAttribute, IntAttribute
from utility import generate_random_string

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
    'prim::Constant',
]
NODE_TYPE_TO_INDEX = {nt: i for i, nt in enumerate(node_type_options)}
NODE_TYPE_TO_INDEX['input'] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX['hidden'] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX['output'] = len(NODE_TYPE_TO_INDEX)

class NodeGene(BaseGene):
    _gene_attributes = [
        StringAttribute('node_type', options=','.join(node_type_options))
    ]
    def __init__(self, node_id: int, node: torch._C.Node=None):
        self.dynamic_attributes = {}
        if node:
            self.node_type = node.kind()
            for attribute_name in node.attributeNames():
                attribute_type = node.kindOf(attribute_name)
                if attribute_type == 'i':
                    attribute = IntAttribute(attribute_name)
                    self.dynamic_attributes[attribute] = node.i(attribute_name)
                elif attribute_type == 'f':
                    attribute = FloatAttribute(attribute_name)
                    self.dynamic_attributes[attribute] = node.f(attribute_name)
                elif attribute_type == 's':
                    attribute = StringAttribute(attribute_name)
                    self.dynamic_attributes[attribute] = node.s(attribute_name)
                else:
                    warn(f'WARNING: Unknown attribute type for node [{node}]: {attribute_type}')
        else:
            self.node_type = None
        # if not hasattr(self, 'value'):
        #     self.value = None
        BaseGene.__init__(self, node_id)

    def copy(self):
        new_gene = NodeGene(self.id)
        new_gene.node_type = self.node_type
        new_gene.dynamic_attributes = self.dynamic_attributes.copy()
        for attribute in self.dynamic_attributes:
            new_gene.dynamic_attributes[attribute] = self.dynamic_attributes[attribute]
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
        if len(self.dynamic_attributes) > 0 and random.random() < config.attribute_delete_prob:
            to_remove = random.choice(self.dynamic_attributes.keys())
            self.remove_attribute(to_remove)

    def add_attribute(self, attr: BaseAttribute, config):
        """Add a new attribute to this gene at runtime."""
        self.dynamic_attributes[attr] = attr.init_value(config)

    def remove_attribute(self, attr_to_remove: BaseAttribute):
        """Remove a dynamically added attribute."""
        self.dynamic_attributes = {a: v for a, v in self.dynamic_attributes.items() if a != attr_to_remove}

    def distance(self, other, config):
        d = 0.0 if self.node_type == other.node_type else 1.0
        for attribute, value in self.dynamic_attributes.items():
            if attribute in other.dynamic_attributes.keys():
                if self.dynamic_attributes[attribute] != other.dynamic_attributes[attribute]:
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
        for a in self.dynamic_attributes:
            if a not in gene2.dynamic_attributes.keys() or random.random() > 0.5:
                new_gene.dynamic_attributes[a] = self.dynamic_attributes[a]
            elif hasattr(gene2, a.name):
                new_gene.dynamic_attributes[a] = gene2.dynamic_attributes[a]
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
