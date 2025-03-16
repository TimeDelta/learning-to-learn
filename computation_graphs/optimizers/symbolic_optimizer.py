import torch
import torch.nn as nn

AVAILABLE_INPUT_SYMBOLS = ['global_loss', 'previous_global_loss', 'current_params']

class SymbolicOptimizer(nn.Module):
    def __init__(self, input_mapping):
        """
        input_mapping [str]: A list of symbolic names defining the inputs that the optimizer expects
        """
        super(SymbolicOptimizer, self).__init__()
        for s in input_mapping:
            if s.lower() not in AVAILABLE_INPUT_SYMBOLS:
                raise Exception('Unknown input symbol: ' + s)
        self.input_mapping = input_mapping

    def compute_symbolic_inputs(self, model, current_params, prev_loss, loss_fn):
        """
        Computes only the symbolic inputs needed by the optimizer, as given by its input_mapping.

        For example, if optimizer.input_mapping is ["current_param", "global_loss", "prev_loss"],
        then only those three values are computed.

        Args:
          prev_loss: The loss computed in the previous iteration.
          loss_fn: A function that accepts the model and returns a scalar loss.

        Returns:
          A dictionary mapping each required key to its computed value.
        """
        symbolic_values = {}
        required = self.input_mapping
        if "current_params" in required:
            symbolic_values["current_params"] = current_params
        if "global_loss" in required:
            symbolic_values["global_loss"] = loss_fn(model)
        if "prev_loss" in required:
            symbolic_values["prev_loss"] = prev_loss
        input_vector = [float(symbolic_values[key]) for key in self.input_mapping]
        return torch.tensor(input_vector, dtype=torch.float32)
