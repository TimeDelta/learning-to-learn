import torch
import torch.jit

import types
import warnings

# copied from neat-python package to add torchscript compatibility since
# neat-python package passes function references

@torch.jit.script
class Product(object):
    def __call__(self, x):
        result = 1.0
        for i in x:
            result *= i
        return result

@torch.jit.script
class Subtract(object):
    def __call__(self, x):
        total = x[0]
        for i in range(1, len(x)):
            total -= x[i]
        return total

@torch.jit.script
class Sum(object):
    def __call__(self, x):
        return sum(x)

@torch.jit.script
class Max(object):
    def __call__(self, x):
        return max(x)

@torch.jit.script
class Min(object):
    def __call__(self, x):
        return min(x)

@torch.jit.script
class Median(object):
    def __call__(self, x):
        return torch.median(x)

@torch.jit.script
class Mean(object):
    def __call__(self, x):
        return torch.mean(x)

class InvalidAggregationFunction(TypeError):
    pass

class AggregationFunctionSet(object):
    """Contains aggregation functions and methods to add and retrieve them."""

    def __init__(self):
        self.functions = {}
        self.add('subtract', Subtract())
        self.add('product', Product())
        self.add('sum', Sum())
        self.add('max', Max())
        self.add('min', Min())
        self.add('median', Median())
        self.add('mean', Mean())

    def add(self, name, function):
        self.functions[name] = function

    def get(self, name):
        f = self.functions.get(name)
        if f is None:
            raise InvalidAggregationFunction("No such aggregation function: {0!r}".format(name))

        return f

    def __getitem__(self, index):
        warnings.warn("Use get, not indexing ([{!r}]), for aggregation functions".format(index),
                      DeprecationWarning)
        return self.get(index)

    def is_valid(self, name):
        return name in self.functions
