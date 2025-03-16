import neat
import math
import random
import copy

def base_model_loss_function(data, model): # model should be a PyTorch model
    model.eval()
    loss = 0.
    for example in data:
        with torch.no_grad():
            actual = model(example[0])
        loss += (actual - expected) ** 2
    return loss

def update_model_params(model, weights, biases, weight_keys, bias_keys):
    """
    Update the model's parameters using its state dict.

    Parameters:
      - model: the PyTorch model.
      - weights: a list of new weight values (as tensors) for the keys in weight_keys.
      - biases: a list of new bias values (as tensors) for the keys in bias_keys.
      - weight_keys: list of state_dict keys corresponding to the weight parameters.
      - bias_keys: list of state_dict keys corresponding to the bias parameters.
    """
    state_dict = model.state_dict()
    for i, key in enumerate(weight_keys):
        state_dict[key] = weights[i]
    for i, key in enumerate(bias_keys):
        state_dict[key] = biases[i]
    model.load_state_dict(state_dict)

def eval_genomes(genomes, config, model, loss_fn,
                 initial_weights, initial_biases, weight_keys, bias_keys,
                 steps=10, epsilon=1e-6):
    """
    Evaluate each genome by using its network as a meta–optimizer that updates both
    the weight and bias parameters of a model via PyTorch's state dict.

    Parameters:
      - genomes: list of (genome_id, genome) tuples
      - config: NEAT configuration
      - model: the PyTorch model to be optimized
      - loss_fn: function accepting the model and returning a scalar loss
      - initial_weights: list of initial weight values (each as a torch.Tensor)
      - initial_biases: list of initial bias values (each as a torch.Tensor)
      - weight_keys: list of state_dict keys for the weights
      - bias_keys: list of state_dict keys for the biases.
      - steps: number of optimization steps.
      - epsilon: a small constant to avoid division by zero.

    For each parameter pair, the meta–optimizer (a NEAT-evolved network) receives an input vector:
        [current_weight, current_bias, global_loss, prev_query_loss]
    and outputs two numbers: [new_weight, new_bias].

    The global loss is computed by updating the model with the full set of parameters.
    For each parameter pair, a temporary update is applied to compute a query loss.

    The final fitness is defined as:
         1.0 / (final_global_loss + average_query_loss + epsilon)
    """
    for genome_id, genome in genomes:
        # Create the meta-optimizer network.
        net = CustomFeedForwardNetwork.create(genome, config)

        # Start with copies of the initial parameters.
        weights = list(initial_weights)
        biases = list(initial_biases)

        # Update the model with the initial parameters.
        update_model_params(model, weights, biases, weight_keys, bias_keys)
        global_loss = loss_fn(model)
        # Initialize previous query losses for each parameter pair.
        prev_query_losses = [global_loss for _ in weights]

        for step in range(steps):
            update_model_params(model, weights, biases, weight_keys, bias_keys)
            global_loss = loss_fn(model)
            new_weights = []
            new_biases = []
            new_prev_query_losses = []

            # Update each parameter pair independently.
            for i, (w, b) in enumerate(zip(weights, biases)):
                # Convert tensor values to Python floats if needed.
                w_val = w.item() if isinstance(w, torch.Tensor) else w
                b_val = b.item() if isinstance(b, torch.Tensor) else b
                input_vector = [w_val, b_val, global_loss, prev_query_losses[i]]
                outputs = net.activate(input_vector)
                # Use the outputs as the new weight and bias values.
                new_w = torch.tensor(outputs[0])
                new_b = torch.tensor(outputs[1])
                new_weights.append(new_w)
                new_biases.append(new_b)

                # Compute the query loss for parameter i by updating just that parameter.
                temp_weights = weights.copy()
                temp_biases = biases.copy()
                temp_weights[i] = new_w
                temp_biases[i] = new_b
                update_model_params(model, temp_weights, temp_biases, weight_keys, bias_keys)
                query_loss = loss_fn(model)
                new_prev_query_losses.append(query_loss)

            weights = new_weights
            biases = new_biases
            prev_query_losses = new_prev_query_losses

        update_model_params(model, weights, biases, weight_keys, bias_keys)
        final_loss = loss_fn(model)
        avg_query_loss = sum(prev_query_losses) / len(prev_query_losses)
        genome.fitness = 1.0 / (final_loss + avg_query_loss + epsilon)

def evaluate_optimizer(optimizer, model, loss_fn, initial_param, steps=10):
    """
    Runs the optimizer over a number of steps.

    Args:
      optimizer: An instance of SymbolicOptimizer (or subclass) that updates a parameter.
      model: The model whose performance is measured by loss_fn.
      loss_fn: Function that takes the model (or a parameter) and returns a scalar loss.
      initial_param: The starting parameter value (assumed scalar for simplicity).
      steps: Number of update iterations.

    Returns:
      The final parameter value and final loss.
    """
    current_param = initial_param
    prev_loss = loss_fn(model)

    for step in range(steps):
        inputs = compute_symbolic_inputs(optimizer, model, current_param, prev_loss, loss_fn)
        new_param = optimizer(inputs, loss_fn, current_param)
        current_param = new_param.item() if isinstance(new_param, torch.Tensor) else new_param
        prev_loss = loss_fn(model)

    final_loss = loss_fn(model)
    return current_param, final_loss

def eval_genomes_wrapper(genomes, config):
    eval_genomes(genomes, config, quadratic_loss, init_param=0.0, steps=10)

# This is a subclass of the neat-python FeedForwardNetwork that we can later extend
# if we need more control over the network’s behavior.
class CustomFeedForwardNetwork(neat.nn.FeedForwardNetwork):
    @classmethod
    def create(cls, genome, config):
        # default create won't work: TODO
        return super(CustomFeedForwardNetwork, cls).create(genome, config)

def create_initial_genome(config):
    """
    Creates a genome with an exact initial network structure.

    In this example we assume a network with 4 inputs (0-3) and 2 outputs (4-5),
    plus one hidden node (6). Inputs 0-3 connect to the hidden node, and the hidden node
    connects to both outputs.
    """
    # Create a new genome with key 0 (the key will be re-assigned later for each population member).
    genome = config.genome_type(0)

    # Manually create nodes.
    # For neat-python, node genes are stored in genome.nodes (a dict mapping node key to node gene).
    # We create nodes with the necessary parameters. Here we assume the defaults for bias, response, etc.
    input_keys = [0, 1, 2, 3]
    output_keys = [4, 5]
    hidden_keys = [6]

    for k in input_keys:
        genome.nodes[k] = config.genome_type.NodeGene(k)
        genome.nodes[k].bias = 0.0
        genome.nodes[k].activation = 'tanh'
        genome.nodes[k].aggregation = 'sum'

    for k in output_keys:
        genome.nodes[k] = config.genome_type.NodeGene(k)
        genome.nodes[k].bias = 0.0
        genome.nodes[k].activation = 'tanh'
        genome.nodes[k].aggregation = 'sum'

    for k in hidden_keys:
        genome.nodes[k] = config.genome_type.NodeGene(k)
        genome.nodes[k].bias = 0.0
        genome.nodes[k].activation = 'tanh'
        genome.nodes[k].aggregation = 'sum'

    # Manually create connections.
    # In neat-python, connections are stored in genome.connections, a dict keyed by a tuple (in_node, out_node).
    # We add connections from each input to the hidden node, then from the hidden node to each output.
    connections = {}
    innovation = 0

    for in_node in input_keys:
        key = (in_node, 6)
        conn = config.genome_type.ConnectionGene(key, 0.0)
        conn.weight = 1.0
        conn.enabled = True
        conn.innovation = innovation
        innovation += 1
        connections[key] = conn

    for out_node in output_keys:
        key = (6, out_node)
        conn = config.genome_type.ConnectionGene(key, 0.0)
        conn.weight = 1.0
        conn.enabled = True
        conn.innovation = innovation
        innovation += 1
        connections[key] = conn

    genome.connections = connections
    return genome

def initialize_population(config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
    if optimizer == 'hill':
        script_module = torch.jit.load('computation_graphs/optimizers/hill_climber.pt')
    elif optimizer == 'gd_backprop':
        script_module = torch.jit.load('computation_graphs/optimizers/gradient_descent_backprop.pt')
    elif optimizer == 'gd_finite_diff':
        script_module = torch.jit.load('gradient_descent_finite_diff.pt')
    # below is a torch._C.Graph object interface is unstable so lock to specific version of PyTorch
    graph = script_module.graph
    for node in graph.nodes():
        print(node)
    base_genome = create_initial_genome(config)
    new_population = {}
    for key in population.population.keys():
        new_genome = copy.deepcopy(base_genome)
        new_genome.key = key
        new_population[key] = new_genome
    population.population = new_population

if __name__ == "__main__":
    # Assumes the configuration file is named 'config-meta'
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         config_file)
    population = neat.Population('neat-config')

    # Override the population with our exact initial genome.
    initialize_population_with_exact_network(population, config)

    # Add reporters.
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Run NEAT for up to 100 generations.
    winner = population.run(eval_genomes_wrapper, 100)
    print('\nBest genome:\n{!s}'.format(winner))
