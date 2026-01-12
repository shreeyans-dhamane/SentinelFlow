import numpy as np
import cv2
from collections import deque
from .interfaces import BaseModule
from .config import SystemConfiguration

class TensorTransformer(BaseModule):
    def __init__(self, config: SystemConfiguration):
        super().__init__("Preprocessor")
        self.cfg = config
        self.temporal_buffer = deque(maxlen=self.cfg.BUFFER_SIZE)
        
    def initialize(self):
        for _ in range(self.cfg.BUFFER_SIZE):
            zeros = np.zeros(self.cfg.input_shape, dtype=np.float32)
            self.temporal_buffer.append(zeros)
        return True

    def _normalize(self, frame):
        frame_resized = cv2.resize(frame, (self.cfg.FRAME_WIDTH, self.cfg.FRAME_HEIGHT))
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        tensor = frame_rgb.astype(np.float32)
        tensor = (tensor / 127.5) - 1.0
        return tensor

    def process(self, raw_frame):
        if raw_frame is None:
            return None

        processed_frame = self._normalize(raw_frame)
        
        self.temporal_buffer.append(processed_frame)
        
        batch_tensor = np.array(self.temporal_buffer)
        batch_tensor = np.expand_dims(batch_tensor, axis=0) 
        
        return batch_tensor

    def shutdown(self):
        self.temporal_buffer.clear()
