from typing import List

class AggregationFunction:
    def __call__(self, inputs: List[float]) -> float:
        raise NotImplementedError(f"Must implement __call__ method for {self} aggregation.")

    def __str__(self) -> str:
        return self.__class__.__name__

class Sum(AggregationFunction):
    def __call__(self, inputs: List[float]) -> float:
        return sum(inputs)

class Product(AggregationFunction):
    def __call__(self, inputs: List[float]) -> float:
        result = 1.0
        for v in inputs:
            result *= v
        return result
