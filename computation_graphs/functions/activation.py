class ActivationFunction:
    def __call__    (self, x: float) -> float:
        raise NotImplementedError("Must implement __call__ method.")

    def __str__(self) -> str:
        return self.__class__.__name__

class Identity(ActivationFunction):
    def __call__    (self, x: float) -> float:
        return x

class Tanh(ActivationFunction):
    def __call__    (self, x: float) -> float:
        return math.tanh(x)

class ReLU(ActivationFunction):
    def __call__    (self, x: float) -> float:
        return x if x > 0 else 0
