from dataclasses import dataclass
from typing import Tuple

@dataclass
class SystemConfiguration:
    
    FRAME_WIDTH: int = 224
    FRAME_HEIGHT: int = 224
    CHANNELS: int = 3
    BUFFER_SIZE: int = 30  
    
    BACKBONE: str = "mobilenet_v2"
    LATENT_DIM: int = 1280
    LSTM_UNITS: int = 128
    DROPOUT_RATE: float = 0.4
    
    Kp: float = 1.2  # Proportional Gain
    Ki: float = 0.1  # Integral Gain
    Kd: float = 0.05 # Derivative Gain
    
    RISK_CRITICAL: float = 0.85
    RISK_RECOVERY: float = 0.25
    ACTUATOR_LATENCY_MS: int = 50
    
    ENABLE_METRICS: bool = True
    LOG_LEVEL: str = "INFO"

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return (self.FRAME_WIDTH, self.FRAME_HEIGHT, self.CHANNELS)
