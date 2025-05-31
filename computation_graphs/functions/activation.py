import math

import torch
import torch.jit

# copied from neat-python package to add torchscript compatibility since
# neat-python package passes function references


@torch.jit.script
class Constant(object):
    def __init__(self, c):
        self.c = c

    def __call__(self, x):
        return self.c


@torch.jit.script
class OneMinus(object):
    def __call__(self, x: float) -> float:
        return 1.0 - x


@torch.jit.script
class Tanh(object):
    def __call__(self, x: float) -> float:
        return math.tanh(x)


@torch.jit.script
class Sigmoid(object):
    def __call__(self, x: float):
        return 1.0 / (1.0 + math.exp(-x))


@torch.jit.script
class Sin(object):
    def __call__(self, x: float):
        return math.sin(x)


@torch.jit.script
class Clamped(object):
    def __call__(self, x: float):
        return max(-1.0, min(1.0, x))


@torch.jit.script
class Inv(object):
    def __call__(self, x: float):
        if x == 0:
            return x
        return 1.0 / x


@torch.jit.script
class Log(object):
    def __call__(self, x: float):
        x = max(1e-10, x)
        return math.log(x)


@torch.jit.script
class Exp(object):
    def __call__(self, x: float):
        x = max(-60.0, min(60.0, x))
        return math.exp(x)


@torch.jit.script
class Abs(object):
    def __call__(self, x: float):
        return abs(x)


@torch.jit.script
class Hat(object):
    def __call__(self, x: float):
        return max(0.0, 1 - abs(x))


@torch.jit.script
class Square(object):
    def __call__(self, x: float):
        return x**2


@torch.jit.script
class Cube(object):
    def __call__(self, x: float):
        return x**3


@torch.jit.script
class LoopActivation:
    def __init__(self, max_iterations: int, block_genes):
        """
        Construct a LoopActivation from a TorchScript prim::Loop node.

        Extracts loop-specific attributes and converts the loop body (the first block)
        into a sub-genome representation.
        """
        self.max_iterations = max_iterations
        self.block_genes = block_genes

    def __call__(self, x: float) -> float:
        for block_gene in self.block_genes:
            # TODO
            print(block_gene)
        return x

    def __str__(self) -> str:
        return f"LoopActivation(max_iter={self.max_iterations}, subgenome={self.block_genes})"


class InvalidActivationFunction(TypeError):
    pass


class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.add("prim::constant", Constant)
        self.add("one_minus", OneMinus())
        # self.add('sigmoid', Sigmoid())
        # self.add('tanh', Tanh())
        # self.add('sin', Sin())
        # self.add('clamped', Clamped())
        # self.add('inv', Inv())
        # self.add('log', Log())
        # self.add('exp', Exp())
        # self.add('abs', Abs())
        # self.add('hat', Hat())
        # self.add('square', Square())
        # self.add('cube', Cube())

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        return self.functions.get(name.lower())

    def is_valid(self, name):
        return name in self.functions
