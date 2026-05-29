"""
RL is a way to tell the model that he is in the right way.
giving rewards for hitting the right stuff
goal: create a dynamic vector of rewards, that if the model hits them in a straight line, a new weight appears, something new gets born.
possibly an emotion.
"""

import enum


class States(enum.IntEnum):
    SAD = 0
    HAPPY = 1
    NORMAL = 3
