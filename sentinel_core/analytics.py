import numpy as np
from .interfaces import BaseModule
from .logger import ResearchLogger

class UncertaintyQuantifier(BaseModule):
    def __init__(self, history_len=20):
        super().__init__("Analytics")
        self.history = []
        self.maxlen = history_len
        self.logger = ResearchLogger()

    def initialize(self):
        return True

    def process(self, raw_risk: float) -> float:
        self.history.append(raw_risk)
        if len(self.history) > self.maxlen:
            self.history.pop(0)

        mu = np.mean(self.history)
        sigma = np.std(self.history)
        
        if sigma > 0.15:
            self.logger.log("DEBUG", f"High Uncertainty (std={sigma:.3f}). Dampening signal.")
            return mu
        
        return raw_risk

    def shutdown(self):
        pass
