import logging
import random
from typing import Dict, Tuple
from warnings import warn

import torch
from neat.attributes import BaseAttribute
from neat.genes import BaseGene

from attributes import BoolAttribute, FloatAttribute, IntAttribute, StringAttribute
from utility import generate_random_string

logger = logging.getLogger(__name__)

# modified from neat-python versions
NODE_TYPE_OPTIONS = [  # for mutation
    "aten::add",
    "aten::sub",
    "aten::mul",
    "aten::div",
    "aten::pow",
    "aten::matmul",
    "aten::transpose",
    "aten::max",
    "aten::min",
    "aten::sum",
    "aten::len",
    "aten::grad",
    "prim::Constant",
    "prim::ListConstruct",
    "prim::DictConstruct",
    "prim::Loop",
    "prim::GetAttr",
    "prim::SetAttr",
    "prim::min",
]
NODE_TYPE_TO_INDEX = {nt: i for i, nt in enumerate(NODE_TYPE_OPTIONS)}
NODE_TYPE_TO_INDEX["input"] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX["hidden"] = len(NODE_TYPE_TO_INDEX)
NODE_TYPE_TO_INDEX["output"] = len(NODE_TYPE_TO_INDEX)
ATTRIBUTE_NAMES = set()


class NodeGene(BaseGene):
    _gene_attributes = [
        StringAttribute("node_type", options=NODE_TYPE_OPTIONS),
        FloatAttribute("attribute_add_prob"),
        FloatAttribute("attribute_delete_prob"),
    ]

    def __init__(self, node_id: int, node: torch._C.Node = None):
        super().__init__(node_id)
        self.dynamic_attributes = {}
        if node is not None:
            self.node_type = node.kind()
            for attribute_name in node.attributeNames():
                attribute_type = node.kindOf(attribute_name)
                if attribute_type == "i":
                    attribute = IntAttribute(attribute_name)
                    self.dynamic_attributes[attribute] = node.i(attribute_name)
                elif attribute_type == "f":
                    attribute = FloatAttribute(attribute_name)
                    self.dynamic_attributes[attribute] = node.f(attribute_name)
                elif attribute_type == "s":
                    attribute = StringAttribute(attribute_name, options=list(ATTRIBUTE_NAMES))
                    self.dynamic_attributes[attribute] = node.s(attribute_name)
                else:
                    warn(f"Unknown attribute type for node [{node}]: {attribute_type}")
                ATTRIBUTE_NAMES.add(attribute_name)
                if not self.dynamic_attributes[attribute]:
                    warn(f"Missing value for " + str(attribute))
            self.num_outputs = len(list(node.outputs()))
            self.output_debug_names = [o.debugName() for o in node.outputs()]
            self.scope = node.scopeName()
        else:
            # placeholder for non-TORCH-NODE instantiation
            self.node_type = None
            self.num_outputs = 0
            self.output_debug_names = []
            self.scope = ""

        logger.debug(
            "NodeGene %s kind=%s, outputs=%s, attrs=%s",
            node_id,
            self.node_type,
            self.num_outputs,
            self.dynamic_attributes,
        )

    def mutate(self, config):
        if random.random() < config.attribute_add_prob:
            r = random.random()
            if r <= 0.25:
                attr = BoolAttribute(generate_random_string(5))
            elif r <= 0.5:
                attr = IntAttribute(generate_random_string(5))
            elif r <= 0.75:
                attr = FloatAttribute(generate_random_string(5))
            else:
                attr = StringAttribute(generate_random_string(5), options=list(ATTRIBUTE_NAMES))
            print(attr)
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

    def crossover(self, other):
        """Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == other.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        new_gene.node_type = self.node_type if random.random() > 0.5 else other.node_type
        all_attrs = set(self.dynamic_attributes) | set(other.dynamic_attributes)
        for attr in all_attrs:
            if attr in self.dynamic_attributes and attr in other.dynamic_attributes:
                parent = self if random.random() > 0.5 else other
                new_gene.dynamic_attributes[attr] = parent.dynamic_attributes[attr]
            elif attr in self.dynamic_attributes:
                new_gene.dynamic_attributes[attr] = self.dynamic_attributes[attr]
            else:
                new_gene.dynamic_attributes[attr] = other.dynamic_attributes[attr]
        return new_gene

    def copy(self):
        new_gene = self.__class__(self.key)
        new_gene.node_type = self.node_type
        new_gene.dynamic_attributes = self.dynamic_attributes
        return new_gene

    def __str__(self):
        return f"NodeGene(id={self.key}, type={self.node_type}, attrs={self.dynamic_attributes})"


class ConnectionGene(BaseGene):
    _gene_attributes = [
        BoolAttribute("enabled"),
    ]

    def __init__(self, key: Tuple[int, int], src_out_idx: int = 0, dst_in_idx: int = 0, param_name: str = None):
        super().__init__(key)
        self.enabled = True
        self.innovation = 0

        # new metadata
        self.src_out_idx = src_out_idx
        self.dst_in_idx = dst_in_idx
        self.param_name = param_name

    def copy(self):
        new_conn = ConnectionGene(
            self.key, src_out_idx=self.src_out_idx, dst_in_idx=self.dst_in_idx, param_name=self.param_name
        )
        new_conn.enabled = self.enabled
        new_conn.innovation = self.innovation
        return new_conn

    def __str__(self):
        return (
            f"ConnectionGene(in={self.in_node}[out{self.src_out_idx}], "
            f"out={self.out_node}[in{self.dst_in_idx}], "
            f"enabled={self.enabled}, innov={self.innovation}, "
            f"param={self.param_name})"
        )

    def distance(self, other, config):
        d = 0.0
        if self.key[0] != other.key[0]:
            d += 1
        if self.key[1] != other.key[1]:
            d += 1
        if self.enabled != other.enabled:
            d += 1
        weight_self = getattr(self, "weight", 0.0)
        weight_other = getattr(other, "weight", 0.0)
        weight_diff = abs(weight_self - weight_other)
        return d * config.compatibility_weight_coefficient + weight_diff
