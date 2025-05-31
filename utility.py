import random
import string


def generate_random_string(length):
    characters = string.ascii_letters
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string
