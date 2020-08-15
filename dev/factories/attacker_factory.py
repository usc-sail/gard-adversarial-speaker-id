from art import attacks
from dev import attacks as attacks_dev

class AttackerFactory():
    def __call__(self, name):
        if name in dir(attacks):
            return getattr(attacks, name)
        elif name in dir(attacks_dev):
            return getattr(attacks_dev, name)
        else:
            raise ValueError(f"Unrecognizable attack type: {name}")

