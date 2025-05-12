from neat.attributes import BaseAttribute, BoolAttribute, StringAttribute, FloatAttribute

from random import choice, gauss, random, uniform
from warnings import warn

"""
Modified from neat-python versions
"""

class FloatAttribute(BaseAttribute):
    """
    Class for numeric attributes,
    such as the response of a node or the weight of a connection.
    """
    _config_items = {"default": [float, 1.0],
                     "init_mean": [float, 0.0],
                     "init_stdev": [float, 1.0],
                     "init_type": [str, 'gaussian'],
                     "replace_rate": [float, .03],
                     "mutate_rate": [float, .03],
                     "mutate_power": [float, .5],
                     "max_value": [float, 1000.0],
                     "min_value": [float, -1000.0]}

    def clamp(self, value, config):
        min_value = getattr(config, self.min_value_name)
        max_value = getattr(config, self.max_value_name)
        return max(min(value, max_value), min_value)

    def init_value(self, config):
        try:
            mean = getattr(config, self.init_mean_name)
        except AttributeError as e:
            mean = 0.0
        try:
            stdev = getattr(config, self.init_stdev_name)
        except AttributeError as e:
            stdev = 1.0
        try:
            init_type = getattr(config, self.init_type_name).lower()
        except AttributeError as e:
            init_type = 'gauss'

        if ('gauss' in init_type) or ('normal' in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if 'uniform' in init_type:
            min_value = max(getattr(config, self.min_value_name), (mean-(2*stdev)))
            max_value = min(getattr(config, self.max_value_name), (mean+(2*stdev)))
            return uniform(min_value, max_value)

        raise RuntimeError("Unknown init_type {!r} for {!s}".format(getattr(config, self.init_type_name), self.init_type_name))

    def validate(self, config): # pragma: no cover
        pass


class IntAttribute(FloatAttribute):
    def clamp(self, value, config):
        return int(super().clamp(value, config))

    def init_value(self, config):
        try:
            val = super().init_value(config)
        except AttributeError as e:
            val = random() * 100
        return int(val)


class BoolAttribute(BaseAttribute):
    """Class for boolean attributes such as whether a connection is enabled or not."""
    _config_items = {"default": [str, True],
                     "mutate_rate": [float, .03],
                     "rate_to_true_add": [float, 0.0],
                     "rate_to_false_add": [float, 0.0]}

    def init_value(self, config):
        try:
            default = str(getattr(config, self.default_name)).lower()
        except AttributeError as e:
            default = '1'

        if default in ('1', 'on', 'yes', 'true'):
            return True
        elif default in ('0', 'off', 'no', 'false'):
            return False
        elif default in ('random', 'none'):
            return bool(random() < 0.5)

        raise RuntimeError("Unknown default value {!r} for {!s}".format(default, self.name))

    def validate(self, config): # pragma: no cover
        pass


class StringAttribute(BaseAttribute):
    """
    Class for string attributes such as the aggregation function of a node,
    which are selected from a list of options.
    """
    _config_items = {"default": [str, 'random'],
                     "options": [list, None],
                     "mutate_rate": [float, .03]}

    def init_value(self, config):
        try:
            default = getattr(config, self.default_name)
        except AttributeError as e:
            default = 'init'

        if default.lower() in ('none','random'):
            options = getattr(config, self.options_name)
            return choice(options)

        return default

    def validate(self, config): # pragma: no cover
        pass
