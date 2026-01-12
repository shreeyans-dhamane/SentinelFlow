import logging
import sys
from datetime import datetime

class ResearchLogger:
    _instance = None

    def __new__(cls, name="SentinelFlow"):
        if cls._instance is None:
            cls._instance = super(ResearchLogger, cls).__new__(cls)
            cls._instance._initialize_logger(name)
        return cls._instance

    def _initialize_logger(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        if not self.logger.handlers:
            formatter = logging.Formatter(
                fmt='[%(asctime)s.%(msecs)03d] [%(levelname)s] [%(module)s]: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            ch = logging.StreamHandler(sys.stdout)
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            fh = logging.FileHandler(f'experiment_{datetime.now().strftime("%Y%m%d")}.log')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def log(self, level: str, message: str):
        if level.upper() == "INFO":
            self.logger.info(message)
        elif level.upper() == "WARNING":
            self.logger.warning(message)
        elif level.upper() == "ERROR":
            self.logger.error(message)
        elif level.upper() == "CRITICAL":
            self.logger.critical(message)
        else:
            self.logger.debug(message)
