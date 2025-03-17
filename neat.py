import neat
import math
import random
import copy
import torch
import torch.nn as nn

from loss import MSELoss

def base_model_loss_function(data, model): # model should be a PyTorch model
    model.eval()
    loss = 0.
    for example in data:
        with torch.no_grad():
            actual = model(example[0])
        loss += (actual - expected) ** 2
    return loss

def eval_genomes(genomes, config, model, loss_fn, initial_params, steps=10, epsilon=1e-10):
    """
    Evaluate each genome by using its network as a meta–optimizer.

    Parameters:
      - genomes: list of (genome_id, genome) tuples
      - config: NEAT configuration

    The final fitness is defined as:
         1.0 / (final_global_loss + average_query_loss + epsilon)
    """
    for genome_id, genome in genomes:
        model_copy = copy.deepcopy(model)
        optimizer = CustomFeedForwardNetwork.create(genome, config)
        evaluate_optimizer(optimizer, model_copy, loss_fn) # TODO dynamically determine num steps based on generation num

        final_loss = loss_fn(model)
        avg_query_loss = sum(prev_query_losses) / len(prev_query_losses)
        genome.fitness = 1.0 / (final_loss + avg_query_loss + epsilon)

def evaluate_optimizer(optimizer, model, loss_fn, steps=10):
    """
    Runs the optimizer over a number of steps.

    Args:
      optimizer: An instance of SymbolicOptimizer (or subclass) that updates a parameter.
      model: The model whose performance is measured by loss_fn.
      loss_fn: Function that takes the model (or a parameter) and returns a scalar loss.
      steps: Number of update iterations.

    Returns:
      1/∑_steps_(∑_dimensions_(loss))
    """
    current_param = initial_param
    prev_loss = loss_fn(model)
    area_under_loss = prev_loss

    for step in range(steps):
        model = optimizer(model, loss_fn, prev_loss)
        prev_loss = loss_fn(model)
        area_under_loss += prev_loss

    return 1 / area_under_loss.sum()

def eval_genomes_wrapper(genomes, config):
    # TODO: load data
    eval_genomes(genomes, config, MSELoss(data), steps=10)

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

def override_initial_population(population, config):
    """
    Overrides the initial genomes in the population with copies of the exact initial genome.
    """
    if optimizer == 'adam_backprop':
        script_module = torch.jit.load('computation_graphs/optimizers/adam_backprop.pt')
    elif optimizer == 'adam_finite_diff':
        script_module = torch.jit.load('computation_graphs/optimizers/adam_finite_diff.pt')
    elif optimizer == 'gd_backprop':
        script_module = torch.jit.load('computation_graphs/optimizers/gradient_descent_backprop.pt')
    elif optimizer == 'gd_finite_diff':
        script_module = torch.jit.load('gradient_descent_finite_diff.pt')
    # below is a torch._C.Graph object. interface is unstable so lock to specific version of PyTorch
    graph = script_module.graph
    for node in graph.nodes():
        # TODO
        print(node)
    base_genome = create_initial_genome(config)
    new_population = {}
    for key in population.population.keys():
        new_genome = copy.deepcopy(base_genome)
        new_genome.key = key
        new_population[key] = new_genome
    population.population = new_population

if __name__ == "__main__":
    config = neat.Config(neat.DefaultGenome,
                         neat.DefaultReproduction,
                         neat.DefaultSpeciesSet,
                         neat.DefaultStagnation,
                         'neat-config')
    population = neat.Population(config)

    override_initial_population(population, config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    num_generations = 1000
    winner = population.run(eval_genomes_wrapper, num_generations)
    print('\nBest genome:\n{!s}'.format(winner))
