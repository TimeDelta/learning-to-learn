from neat.genome import DefaultGenome


class Genome(DefaultGenome):

    activation_functions = [Identity(), Tanh(), ReLU()]
    aggregation_functtions = [Sum(), Product(), Max(), Min()]

    def __init__(self):
        # Dictionary mapping node id to NodeGene.
        self.nodes: Dict[int, NodeGene] = {}
        # List of connections: each connection is a tuple (in_node_id, out_node_id, weight)
        self.connections: List[Tuple[int, int, float]] = []
        self.next_node_id = 0

    def add_node(self, node_type: str,
                 activation: ActivationFunction = TanhActivation(),
                 aggregation: AggregationFunction = SumAggregation()) -> NodeGene:
        node = NodeGene(self.next_node_id, node_type, activation, aggregation)
        self.nodes[self.next_node_id] = node
        self.next_node_id += 1
        return node

    def add_connection(self, in_id: int, out_id: int, weight: float = None):
        if weight is None:
            weight = random.uniform(-1, 1)
        self.connections.append((in_id, out_id, weight))

    def crossover(self, other: 'Genome') -> 'Genome':
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

    def distance(self, other: 'Genome') -> float:
        # TODO
        # modify this code from superclass's distance method to do partial disjoint
        # based on how many things are different in the the closest node (maybe
        # implement Node.distance(other_node))

        # Compute node gene distance component
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
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

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
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def forward(self, input_values: Dict[int, float]) -> Dict[int, float]:
        """
        A minimal feed-forward evaluation of the genome.
        This implementation assumes a feed-forward network with no recurrent connections.
        'input_values' is a dictionary mapping input node id to its value.
        Returns a dictionary mapping node id to computed output.
        """
        outputs: Dict[int, float] = {}
        # Set outputs for input nodes.
        for node_id, node in self.nodes.items():
            if node.node_type == "input":
                outputs[node_id] = input_values.get(node_id, 0.0)
        unresolved = True
        # Continue until all nodes have computed outputs.
        while unresolved:
            unresolved = False
            for node_id, node in self.nodes.items():
                if node_id in outputs:
                    continue
                # Find incoming connections for which the source output is known.
                incoming = [(in_id, weight) for (in_id, out_id, weight) in self.connections if out_id == node_id]
                if any(in_id not in outputs for in_id, _ in incoming):
                    unresolved = True
                    continue
                weighted_inputs = [outputs[in_id] * weight for in_id, weight in incoming]
                agg = node.aggregate(weighted_inputs)
                outputs[node_id] = node.activate(agg)
        return outputs

    def __str__(self):
        return f"Genome(nodes={self.nodes}, connections={self.connections})"