from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseModule(ABC):
    def __init__(self, name: str):
        self.module_name = name
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize resources (models, buffers, hardware connection)."""
        pass

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Core execution logic for the pipeline step."""
        pass
        
    @abstractmethod
    def shutdown(self):
        """Release resources safely."""
        pass

    def get_metadata(self) -> Dict[str, str]:
        return {"module": self.module_name, "status": "active"}
