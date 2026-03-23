import copy
import hashlib
import logging
import random
from typing import Dict, Optional, Set, Tuple
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
    "tensor",
]
NODE_TYPE_TO_INDEX = {nt: i for i, nt in enumerate(NODE_TYPE_OPTIONS)}
ATTRIBUTE_NAMES: Set[str] = set()
ATTRIBUTE_NAMES_BY_KIND: Dict[str, Set[str]] = {}
ATTRIBUTE_NAMES_VERSION = 0
ATTRIBUTE_VALUE_KINDS_BY_KIND: Dict[Optional[str], Dict[str, Set[str]]] = {}
_SCHEMA_ARGUMENT_NAME_CACHE: Dict[str, Dict[str, str]] = {}
_ALL_SCHEMA_ARGUMENTS: Optional[Dict[str, Dict[str, str]]] = None


def _value_kind_from_schema_type(type_obj) -> Optional[str]:
    if type_obj is None:
        return None
    kind = None
    try:
        kind = type_obj.kind()
    except AttributeError:
        kind = None
    if kind == "OptionalType" and hasattr(type_obj, "getElementType"):
        return _value_kind_from_schema_type(type_obj.getElementType())
    if kind == "ListType" and hasattr(type_obj, "getElementType"):
        inner = _value_kind_from_schema_type(type_obj.getElementType())
        return f"list[{inner or 'any'}]"
    mapping = {
        "TensorType": "tensor",
        "NumberType": "float",
        "FloatType": "float",
        "IntType": "int",
        "BoolType": "bool",
        "StringType": "string",
        "ScalarType": "float",
        "DeviceObjType": "string",
    }
    if kind in mapping:
        return mapping[kind]
    type_str = str(type_obj)
    if type_str.endswith("?"):
        return _value_kind_from_schema_type(type_str[:-1])
    if type_str.startswith("List[") and type_str.endswith("]"):
        inner = _value_kind_from_schema_type(type_str[5:-1])
        return f"list[{inner or 'any'}]"
    if type_str in {"Tensor", "Tensor[]"}:
        return "tensor"
    if type_str in {"int", "int64", "Index"}:
        return "int"
    if type_str in {"float", "Scalar"}:
        return "float"
    if type_str in {"bool"}:
        return "bool"
    if type_str in {"str", "string"}:
        return "string"
    return None


def _value_kind_from_python_value(value) -> Optional[str]:
    if torch.is_tensor(value):
        return "tensor"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (list, tuple)):
        if not value:
            return "list"
        inner = _value_kind_from_python_value(value[0])
        return f"list[{inner or 'any'}]"
    if isinstance(value, dict):
        return "dict"
    return None


def register_attribute_name(
    node_type: str | None,
    attribute_name: str | None,
    value_kind: str | None = None,
) -> None:
    """Record that a node kind exposes a particular attribute name/type."""
    if not attribute_name:
        return
    ATTRIBUTE_NAMES.add(attribute_name)
    key = node_type or None
    if node_type:
        names = ATTRIBUTE_NAMES_BY_KIND.setdefault(node_type, set())
        if attribute_name not in names:
            names.add(attribute_name)
            global ATTRIBUTE_NAMES_VERSION
            ATTRIBUTE_NAMES_VERSION += 1
    if value_kind:
        kind_map = ATTRIBUTE_VALUE_KINDS_BY_KIND.setdefault(key, {})
        kind_set = kind_map.setdefault(attribute_name, set())
        kind_set.add(value_kind)


def canonical_attribute_names_for_kind(node_type: str | None) -> Set[str]:
    if not node_type:
        return set()
    return ATTRIBUTE_NAMES_BY_KIND.get(node_type, set())


def attribute_value_kinds_for_kind(node_type: str | None) -> Dict[str, Set[str]]:
    return ATTRIBUTE_VALUE_KINDS_BY_KIND.get(node_type or None, {})


def attribute_value_kind(node_type: str | None, attribute_name: str) -> Optional[str]:
    kinds = attribute_value_kinds_for_kind(node_type).get(attribute_name)
    if kinds:
        # Prefer deterministic ordering for reproducibility.
        for choice in ("tensor", "float", "int", "bool", "string"):
            if choice in kinds:
                return choice
        return sorted(kinds)[0]
    # fall back to globally registered names
    global_kinds = attribute_value_kinds_for_kind(None).get(attribute_name)
    if global_kinds:
        return sorted(global_kinds)[0]
    return None


def attribute_value_kind_for_index(node_type_idx: int, attribute_name: str) -> Optional[str]:
    node_type = None
    if 0 <= node_type_idx < len(NODE_TYPE_OPTIONS):
        node_type = NODE_TYPE_OPTIONS[node_type_idx]
    return attribute_value_kind(node_type, attribute_name)


def _ensure_schema_argument_map_loaded() -> None:
    global _ALL_SCHEMA_ARGUMENTS
    if _ALL_SCHEMA_ARGUMENTS is not None:
        return
    mapping: Dict[str, Dict[str, str]] = {}
    try:
        schemas = torch._C._jit_get_all_schemas()
    except AttributeError:
        _ALL_SCHEMA_ARGUMENTS = {}
        return
    for schema in schemas:
        base = getattr(schema, "name", None)
        if not base:
            continue
        entry = mapping.setdefault(base, {})
        for argument in getattr(schema, "arguments", []):
            name = getattr(argument, "name", None)
            if name:
                entry.setdefault(name, _value_kind_from_schema_type(argument.type))
        for result in getattr(schema, "returns", []):
            name = getattr(result, "name", None)
            if name:
                entry.setdefault(name, _value_kind_from_schema_type(result.type))
    _ALL_SCHEMA_ARGUMENTS = mapping


def _schema_argument_names_for_kind(node_kind: str) -> Dict[str, str]:
    cached = _SCHEMA_ARGUMENT_NAME_CACHE.get(node_kind)
    if cached is not None:
        return cached
    _ensure_schema_argument_map_loaded()
    names = dict(_ALL_SCHEMA_ARGUMENTS.get(node_kind, {})) if _ALL_SCHEMA_ARGUMENTS is not None else {}
    _SCHEMA_ARGUMENT_NAME_CACHE[node_kind] = names
    return names


def register_schema_argument_names(node: torch._C.Node) -> None:
    if node is None:
        return
    node_kind = node.kind()
    names = _schema_argument_names_for_kind(node_kind)
    for name, value_kind in names.items():
        register_attribute_name(node_kind, name, value_kind)


register_attribute_name(None, "pin_role", "string")
register_attribute_name(None, "pin_slot_index", "int")
register_attribute_name(None, "is_input_pin", "bool")
register_attribute_name(None, "is_output_pin", "bool")


def node_type_name_from_index(index: int) -> str:
    if 0 <= index < len(NODE_TYPE_OPTIONS):
        return NODE_TYPE_OPTIONS[index]
    raise KeyError(f"Unknown node type index {index}")


def node_type_index_from_name(name: str) -> int:
    if not name:
        raise KeyError("Empty node type name")
    if name not in NODE_TYPE_TO_INDEX:
        raise KeyError(f"Unknown node type {name!r}")
    return NODE_TYPE_TO_INDEX[name]


def ensure_node_type_registered(name: str) -> int:
    """Ensure a TorchScript node kind has a stable index, extending vocab if needed."""
    if not name:
        raise KeyError("Empty node type name")
    idx = NODE_TYPE_TO_INDEX.get(name)
    if idx is not None:
        return idx
    idx = len(NODE_TYPE_OPTIONS)
    NODE_TYPE_TO_INDEX[name] = idx
    NODE_TYPE_OPTIONS.append(name)
    return idx


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
            register_schema_argument_names(node)
            for attribute_name in node.attributeNames():
                attribute_type = node.kindOf(attribute_name)
                if attribute_type == "i":
                    attribute = IntAttribute(attribute_name)
                    value = node.i(attribute_name)
                    self.dynamic_attributes[attribute] = value
                elif attribute_type == "f":
                    attribute = FloatAttribute(attribute_name)
                    value = node.f(attribute_name)
                    self.dynamic_attributes[attribute] = value
                elif attribute_type == "s":
                    attribute = StringAttribute(attribute_name, options=list(ATTRIBUTE_NAMES))
                    value = node.s(attribute_name)
                    self.dynamic_attributes[attribute] = value
                else:
                    warn(f"Unknown attribute type for node [{node}]: {attribute_type}")
                    value = None
                register_attribute_name(self.node_type, attribute_name, _value_kind_from_python_value(value))
                if self.dynamic_attributes[attribute] is None:
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
            self.add_attribute(attr, config)

        if len(self.dynamic_attributes) > 0 and random.random() < config.attribute_delete_prob:
            to_remove = random.choice(list(self.dynamic_attributes.keys()))
            self.remove_attribute(to_remove)

    def add_attribute(self, attr: BaseAttribute, config):
        """Add a new attribute to this gene at runtime."""
        max_attrs = getattr(config, "max_attributes_per_node", None)
        if max_attrs is not None:
            try:
                max_attrs = int(max_attrs)
            except (TypeError, ValueError):
                max_attrs = None
        if max_attrs is not None and max_attrs > 0 and len(self.dynamic_attributes) >= max_attrs:
            logger.debug(
                "Skipping attribute add for node %s; already has %d attributes (limit=%d)",
                self.key,
                len(self.dynamic_attributes),
                max_attrs,
            )
            return False
        value = attr.init_value(config)
        self.dynamic_attributes[attr] = value
        register_attribute_name(
            getattr(self, "node_type", None),
            getattr(attr, "name", None),
            _value_kind_from_python_value(value),
        )
        return True

    def remove_attribute(self, attr_to_remove: BaseAttribute):
        """Remove a dynamically added attribute."""
        del self.dynamic_attributes[attr_to_remove]

    def distance(self, other, config):
        def attr_equal(a, b):
            if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
                if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                    return False  # tensors never equal non-tensors
                if a.shape != b.shape:
                    return False
                return torch.allclose(a, b)
            return a == b

        d = 0.0 if self.node_type == other.node_type else 1.0

        common = set(self.dynamic_attributes) & set(other.dynamic_attributes)
        for name in common:
            if not attr_equal(self.dynamic_attributes[name], other.dynamic_attributes[name]):
                d += 1

        # penalty for attrs only in one gene
        d += len(set(self.dynamic_attributes) ^ set(other.dynamic_attributes))

        if getattr(self, "scope", None) != getattr(other, "scope", None):
            d += 1

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
        new_gene.dynamic_attributes = copy.deepcopy(self.dynamic_attributes)
        if hasattr(self, "output_debug_names"):
            new_gene.output_debug_names = copy.deepcopy(self.output_debug_names)
        if hasattr(self, "scope"):
            new_gene.scope = copy.deepcopy(self.scope)
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
