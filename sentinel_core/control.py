=from .interfaces import BaseModule
from .config import SystemConfiguration
from .logger import ResearchLogger

class AdaptiveFlowPID(BaseModule):
    def __init__(self, config: SystemConfiguration):
        super().__init__("PID_Controller")
        self.cfg = config
        self.integral = 0.0
        self.prev_error = 0.0
        self.logger = ResearchLogger()
        self.lockout = False

    def initialize(self):
        self.logger.log("INFO", f"PID Controller Active. Kp={self.cfg.Kp}, Ki={self.cfg.Ki}, Kd={self.cfg.Kd}")
        return True

    def process(self, risk_level: float) -> float:
        if risk_level > self.cfg.RISK_CRITICAL:
            self.lockout = True
        elif risk_level < self.cfg.RISK_RECOVERY:
            self.lockout = False

        if self.lockout:
            return 0.0 

        error = 0.0 - risk_level 
        
        P = self.cfg.Kp * error
        self.integral += error
        I = self.cfg.Ki * self.integral
        D = self.cfg.Kd * (error - self.prev_error)
        
        control_signal = P + I + D
        self.prev_error = error
        flow_output = 100.0 + (control_signal * 100.0)
        
        return max(0.0, min(100.0, flow_output))

    def shutdown(self):
        pass
