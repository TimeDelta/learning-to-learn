import math
import torch
import torch.jit

# copied from neat-python package to add torchscript compatibility since
# neat-python package passes function references

@torch.jit.script
class Tanh(object):
    def __call__(self, x: float) -> float:
        return math.tanh(x)

@torch.jit.script
class Sigmoid(object):
    def __call__(self, x:float):
        return 1.0 / (1.0 + math.exp(-x))

@torch.jit.script
class Sin(object):
    def __call__(self, x:float):
        return math.sin(x)

@torch.jit.script
class Clamped(object):
    def __call__(self, x:float):
        return max(-1.0, min(1.0, x))

@torch.jit.script
class Inv(object):
    def __call__(self, x:float):
        if x == 0:
            return x
        return 1.0 / x

@torch.jit.script
class Log(object):
    def __call__(self, x:float):
        x = max(1e-10, x)
        return math.log(x)

@torch.jit.script
class Exp(object):
    def __call__(self, x:float):
        x = max(-60.0, min(60.0, x))
        return math.exp(x)

@torch.jit.script
class Abs(object):
    def __call__(self, x:float):
        return abs(x)

@torch.jit.script
class Hat(object):
    def __call__(self, x:float):
        return max(0.0, 1 - abs(x))

@torch.jit.script
class Square(object):
    def __call__(self, x:float):
        return x ** 2

@torch.jit.script
class Cube(object):
    def __call__(self, x:float):
        return x ** 3

class InvalidActivationFunction(TypeError):
    pass

class ActivationFunctionSet(object):
    """
    Contains the list of current valid activation functions,
    including methods for adding and getting them.
    """

    def __init__(self):
        self.functions = {}
        self.add('sigmoid', Sigmoid())
        self.add('tanh', Tanh())
        self.add('sin', Sin())
        self.add('clamped', Clamped())
        self.add('inv', Inv())
        self.add('log', Log())
        self.add('exp', Exp())
        self.add('abs', Abs())
        self.add('hat', Hat())
        self.add('square', Square())
        self.add('cube', Cube())

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidActivationFunction("No such activation function: {0!r}".format(name))

        return f

    def is_valid(self, name):
        return name in self.functions
