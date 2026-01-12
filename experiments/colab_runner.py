import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentinel_core.config import SystemConfiguration
from sentinel_core.logger import ResearchLogger
from sentinel_core.video_io import ThreadedVideoInterface
from sentinel_core.preprocessing import TensorTransformer
from sentinel_core.perception import FeatureExtractionFactory
from sentinel_core.cognition import TemporalReasoningUnit
from sentinel_core.analytics import UncertaintyQuantifier
from sentinel_core.control import AdaptiveFlowPID
from sentinel_core.telemetry import PerformanceMonitor

def main(video_path):
    config = SystemConfiguration()
    logger = ResearchLogger("Sentinel_Main")
    
    stream = ThreadedVideoInterface(source=video_path, queue_size=config.BUFFER_SIZE)
    prep = TensorTransformer(config)
    eye = FeatureExtractionFactory()
    brain = TemporalReasoningUnit()
    stats = UncertaintyQuantifier()
    controller = AdaptiveFlowPID(config)
    monitor = PerformanceMonitor()

    modules = [stream, prep, eye, brain, stats, controller, monitor]

    for mod in modules:
        if not mod.initialize(): return

    try:
        while True:
            monitor.tick_start()
            raw = stream.process()
            if raw is None: continue
            
            tensor = prep.process(raw)
            if tensor is None: continue
            
            risk = brain.process(eye.process(tensor))
            valid_risk = stats.process(risk)
            pwm = controller.process(valid_risk)
            
            monitor.tick_end()
            tel = monitor.process()
            logger.log("INFO", f"FPS: {tel['fps']:.1f} | Risk: {valid_risk:.3f} | Flow: {pwm:.1f}%")

    except KeyboardInterrupt:
        pass
    finally:
        for mod in modules: mod.shutdown()

if __name__ == "__main__":
    main("../data/river_surveillance_J26.mp4")
