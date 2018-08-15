from abc import ABC, abstractmethod


class Scorer(ABC):
    """
    Base class for scorers.
    """

    @abstractmethod
    def __call__(self, gts, res):
        pass
