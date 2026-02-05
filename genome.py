import copy
import sys
from itertools import count
from random import choice, random, shuffle
from typing import Dict, List, Tuple

import torch
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.graphs import creates_cycle, required_for_output

from attributes import BoolAttribute, FloatAttribute, IntAttribute, StringAttribute
from computation_graphs.functions.activation import *
from computation_graphs.functions.aggregation import *
from genes import NODE_TYPE_TO_INDEX, ConnectionGene, NodeGene


class OptimizerGenomeConfig(object):
    """
    Copied from DefaultGenomeConfig then modified because it had too many required config params we don't need
    """

    def __init__(self, params):
        self.activation_function_defs = ActivationFunctionSet()
        self.aggregation_function_defs = AggregationFunctionSet()

        self._params = [
            ConfigParameter("compatibility_disjoint_coefficient", float),
            ConfigParameter("compatibility_weight_coefficient", float),
            ConfigParameter("attribute_add_prob", float),
            ConfigParameter("attribute_delete_prob", float),
            ConfigParameter("conn_add_prob", float),
            ConfigParameter("conn_delete_prob", float),
            ConfigParameter("node_add_prob", float),
            ConfigParameter("node_delete_prob", float),
            ConfigParameter("single_structural_mutation", bool, "false"),
            ConfigParameter("structural_mutation_surer", str, "default"),
        ]

        self.num_inputs = 3
        self.num_outputs = 1

        self.node_gene_type = params["node_gene_type"]
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params["connection_gene_type"]
        self._params += self.connection_gene_type.get_config_params()

        for p in self._params:
            value = p.interpret(params)
            print(f"setting {p.name} to {value}")
            setattr(self, p.name, value)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        if self.structural_mutation_surer.lower() in ["1", "yes", "true", "on"]:
            self.structural_mutation_surer = "true"
        elif self.structural_mutation_surer.lower() in ["0", "no", "false", "off"]:
            self.structural_mutation_surer = "false"
        elif self.structural_mutation_surer.lower() == "default":
            self.structural_mutation_surer = "default"
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_function_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        write_pretty_params(f, self, self._params)

    def get_new_node_key(self, node_dict):
        if node_dict:
            return max(node_dict) + 1
        return 0

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == "true":
            return True
        elif self.structural_mutation_surer == "false":
            return False
        elif self.structural_mutation_surer == "default":
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)


class OptimizerGenome(object):
    """
    A genome for generalized optimizers.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection. This connection is permanently
           enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict["node_gene_type"] = NodeGene
        param_dict["connection_gene_type"] = ConnectionGene
        return OptimizerGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        self.key = key

        # (gene_key, gene) pairs for gene sets
        self.nodes: Dict[int, NodeGene] = {}
        self.connections: Dict[int, int] = {}  # [from, to]
        self.next_node_id = 0

        self.fitness = None
        self.fitnesses = []
        self.optimizer = None
        self.optimizer_path = None
        self.graph_dict = None
        self.serialized_module: bytes | None = None

    def __deepcopy__(self, memo):
        # Create a blank instance
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new

        # Deepcopy everything except 'optimizer'
        for k, v in self.__dict__.items():
            if k == "optimizer":
                setattr(new, k, torch.jit.load(self.optimizer_path))
            else:
                setattr(new, k, copy.deepcopy(v, memo))
        return new

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

    def configure_crossover(self, genome1, genome2, config):
        """Configure a new genome by crossover from two parent genomes."""
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """Mutates this genome."""
        # Any structural mutation invalidates previously serialized TorchScript payloads.
        self.serialized_module = None

        if config.single_structural_mutation:
            div = max(
                1, (config.node_add_prob + config.node_delete_prob + config.conn_add_prob + config.conn_delete_prob)
            )
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob + config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < (
                (config.node_add_prob + config.node_delete_prob + config.conn_add_prob + config.conn_delete_prob) / div
            ):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, True)
        self.add_connection(config, new_node_id, o, True)

    def add_connection(self, config, input_key, output_key, enabled):
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = [n for n in self.nodes if n not in config.input_keys]
        if not possible_outputs:
            return

        possible_inputs = list(self.nodes) + config.input_keys
        if not possible_inputs:
            return

        out_node = choice(possible_outputs)
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        # if config.feed_forward and creates_cycle(list(self.connections), key):
        #     return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + (config.compatibility_disjoint_coefficient * disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (
                connection_distance + (config.compatibility_disjoint_coefficient * disjoint_connections)
            ) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        for k, ng in self.nodes.items():
            s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = NodeGene(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(
            self.nodes, self.connections, genome_config.input_keys, genome_config.output_keys
        )
        new_genome = OptimizerGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome

    def compile_optimizer(self, genome_config):
        """Compile this genome into a TorchScript optimizer."""
        from graph_builder import rebuild_and_script

        if self.graph_dict is None:
            node_ids = sorted(self.nodes.keys())
            node_types = []
            node_attributes = []
            for nid in node_ids:
                node = self.nodes[nid]
                idx = NODE_TYPE_TO_INDEX.get(node.node_type)
                if idx is None:
                    raise KeyError(f"Unknown node_type {node.node_type!r}")
                node_types.append(idx)
                node_attributes.append(node.dynamic_attributes)
            node_types = torch.tensor(node_types, dtype=torch.long)

            edges = []
            for (src, dst), conn in self.connections.items():
                if conn.enabled and src in node_ids and dst in node_ids:
                    local_src = node_ids.index(src)
                    local_dst = node_ids.index(dst)
                    edges.append([local_src, local_dst])
            if edges:
                edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            self.graph_dict = {
                "node_types": node_types,
                "edge_index": edge_index,
                "node_attributes": node_attributes,
            }
            if self.serialized_module is not None:
                self.graph_dict["serialized_module"] = self.serialized_module

        self.optimizer = rebuild_and_script(self.graph_dict, genome_config, key=self.key)
        self.optimizer_path = None

    def add_node(self, node_type: str, activation, aggregation) -> NodeGene:
        if activation is None and aggregation is None:
            print("WARNING: node added without any operation")
        node = NodeGene(self.next_node_id, node_type, activation, aggregation)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node

    def crossover(self, other: "Genome") -> "Genome":
        # TODO find better crossover method than this code from the superclass
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def __str__(self):
        return f"Genome(nodes={self.nodes}, connections={self.connections})"


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes
