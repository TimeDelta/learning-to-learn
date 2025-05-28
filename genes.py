from neat.attributes import BaseAttribute
from neat.genes import BaseGene
from typing import Dict
import torch

import random
from warnings import warn

from attributes import BoolAttribute, StringAttribute, FloatAttribute, IntAttribute
from utility import generate_random_string

# modified from neat-python versions
NODE_TYPE_OPTIONS = [ # for mutation
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
    'aten::grad',
    'prim::Constant',
    'prim::ListConstruct',
    'prim::DictConstruct',
    'prim::Loop',
    'prim::GetAttr',
    'prim::SetAttr',
    'prim::min',
]
NODE_TYPE_TO_INDEX = {nt: i for i, nt in enumerate(NODE_TYPE_OPTIONS)}
NODE_TYPE_TO_INDEX['input'] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX['hidden'] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX['output'] = len(NODE_TYPE_TO_INDEX)
ATTRIBUTE_NAMES = set()

class NodeGene(BaseGene):
    _gene_attributes = [
        StringAttribute('node_type', options=','.join(NODE_TYPE_OPTIONS)),
        FloatAttribute('attribute_add_prob'),
        FloatAttribute('attribute_delete_prob'),
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
                    attribute = StringAttribute(attribute_name, options=','.join(ATTRIBUTE_NAMES))
                    self.dynamic_attributes[attribute] = node.s(attribute_name)
                else:
                    warn(f'WARNING: Unknown attribute type for node [{node}]: {attribute_type}')
                ATTRIBUTE_NAMES.add(attribute_name)
                if not self.dynamic_attributes[attribute]:
                    warn(f'Missing value for ' + str(attribute))
        else:
            self.node_type = None

        # DEBUG: see *this* gene's attrs, not the global set
        print(f"NodeGene {node_id} attrs:", self.dynamic_attributes)
        super().__init__(node_id)

    def mutate(self, config):
        if random.random() < config.attribute_add_prob:
            r = random.random()
            if r <= .25:
                attr = BoolAttribute(generate_random_string(5)).init_value(config)
            elif r <= .5:
                attr = IntAttribute(generate_random_string(5)).init_value(config)
            elif r <= .75:
                attr = FloatAttribute(generate_random_string(5)).init_value(config)
            else:
                attr = StringAttribute(generate_random_string(5), options=','.join(ATTRIBUTE_NAMES)).init_value(config)
            self.add_attribute(attr, config)

        if len(self.dynamic_attributes) > 0 and random.random() < config.attribute_delete_prob:
            to_remove = random.choice(list(self.dynamic_attributes.keys()))
            self.remove_attribute(to_remove)

    def add_attribute(self, attr: BaseAttribute, config):
        """Add a new attribute to this gene at runtime."""
        self.dynamic_attributes[attr] = attr.init_value(config)

    def remove_attribute(self, attr_to_remove: BaseAttribute):
        """Remove a dynamically added attribute."""
        del self.dynamic_attributes[attr_to_remove]

    def distance(self, other, config):
        d = 0.0 if self.node_type == other.node_type else 1.0

        common = set(self.dynamic_attributes) & set(other.dynamic_attributes)
        for name in common:
            if self.dynamic_attributes[name] != other.dynamic_attributes[name]:
                d += 1

        # penalty for attrs only in one gene
        d += len(set(self.dynamic_attributes) ^ set(other.dynamic_attributes))

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
                new_gene.dynamic_attributes[name] = other.dynamic_attributes[name]
        return new_gene

    def __str__(self):
        return f"NodeGene(id={self.key}, type={self.node_type}, attrs={self.dynamic_attributes})"

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
