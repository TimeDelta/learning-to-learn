from random import choice, gauss, randint, random, uniform
from warnings import warn

import neat

"""
Modified from neat-python versions
"""


class FloatAttribute(neat.attributes.FloatAttribute):
    _config_items = {
        "default": [float, 1.0],
        "init_mean": [float, 0.0],
        "init_stdev": [float, 1.0],
        "init_type": [str, "gaussian"],
        "replace_rate": [float, 0.03],
        "mutate_rate": [float, 0.03],
        "mutate_power": [float, 0.5],
        "max_value": [float, 1000.0],
        "min_value": [float, -1000.0],
    }

    def clamp(self, value, config):
        return max(min(value, self.max_value), self.min_value)

    def init_value(self, config):
        if hasattr(config, self.init_mean_name):
            mean = getattr(config, self.init_mean_name)
        else:
            mean = 0.0
        if hasattr(config, self.init_stdev_name):
            stdev = getattr(config, self.init_stdev_name)
        else:
            stdev = 10.0
        if hasattr(config, self.init_type_name):
            init_type = getattr(config, self.init_type_name).lower()
        else:
            init_type = "gauss"

        if hasattr(config, self.min_value_name):
            self.min_value = max(getattr(config, self.min_value_name), (mean - (2 * stdev)))
        else:
            self.min_value = mean - (2 * stdev)
        if hasattr(config, self.max_value_name):
            self.max_value = min(getattr(config, self.max_value_name), (mean + (2 * stdev)))
        else:
            self.max_value = mean + (2 * stdev)

        if ("gauss" in init_type) or ("normal" in init_type):
            return self.clamp(gauss(mean, stdev), config)

        if "uniform" in init_type:
            return uniform(self.min_value, self.max_value)

        raise RuntimeError(
            "Unknown init_type {!r} for {!s}".format(getattr(config, self.init_type_name), self.init_type_name)
        )

    def mutate_value(self, value, config):
        # mutate_rate is usually no lower than replace_rate, and frequently higher -
        # so put first for efficiency
        if hasattr(config, self.mutate_rate_name):
            mutate_rate = getattr(config, self.mutate_rate_name)
        else:
            mutate_rate = 0.025  # TODO: turn all of these defaults into hyperparams

        r = random()
        if r < mutate_rate:
            mutate_power = getattr(config, self.mutate_power_name)
            return self.clamp(value + gauss(0.0, mutate_power), config)

        replace_rate = getattr(config, self.replace_rate_name)

        if r < replace_rate + mutate_rate:
            return self.init_value(config)

        return value

    def validate(self, config):  # pragma: no cover
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class IntAttribute(neat.attributes.IntegerAttribute):
    def clamp(self, value, config):
        min_v = getattr(config, self.min_value_name, -1000)
        max_v = getattr(config, self.max_value_name, 1000)
        return int(max(min(value, max_v), min_v))

    def init_value(self, config):
        min_v = getattr(config, self.min_value_name, 0)
        max_v = getattr(config, self.max_value_name, 10)
        return randint(min_v, max_v)

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class BoolAttribute(neat.attributes.BoolAttribute):
    _config_items = {
        "default": [str, True],
        "mutate_rate": [float, 0.03],
        "rate_to_true_add": [float, 0.0],
        "rate_to_false_add": [float, 0.0],
    }

    def init_value(self, config):
        if hasattr(config, self.default_name):
            default = str(getattr(config, self.default_name)).lower()
        else:
            default = "1"

        if default in ("1", "on", "yes", "true"):
            return True
        elif default in ("0", "off", "no", "false"):
            return False
        elif default in ("random", "none"):
            return bool(random() < 0.5)

        raise RuntimeError("Unknown default value {!r} for {!s}".format(default, self.name))

    def mutate_value(self, value, config):
        if hasattr(config, self.mutate_rate_name):
            mutate_rate = getattr(config, self.mutate_rate_name)
        else:
            mutate_rate = 0.025

        if mutate_rate > 0:
            r = random()
            if r < mutate_rate:
                # NOTE: we choose a random value here so that the mutation rate has the
                # same exact meaning as the rates given for the string and bool
                # attributes (the mutation operation *may* change the value but is not
                # guaranteed to do so).
                return random() < 0.5

        return value

    def validate(self, config):  # pragma: no cover
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"


class StringAttribute(neat.attributes.StringAttribute):
    _config_items = {"default": [str, "random"], "options": [list, None], "mutate_rate": [float, 0.03]}

    def __init__(self, name: str, options=None, **default_dict):
        if options is not None:
            default_dict["options"] = options
        super().__init__(name, **default_dict)
        self.options = options or []

    def init_value(self, config):
        options = getattr(config, self.options_name, getattr(self, "options", []))
        default = getattr(config, self.default_name, "random")
        if str(default).lower() in ("none", "random"):
            if options:
                return choice(options)
            return ""
        return default

    def mutate_value(self, value, config):
        mutate_rate = getattr(config, self.mutate_rate_name, 0.025)

        if mutate_rate > 0:
            if random() < mutate_rate:
                options = getattr(config, self.options_name, getattr(self, "options", []))
                if options:
                    return choice(options)
                return value

        return value

    def validate(self, config):  # pragma: no cover
        pass

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"
