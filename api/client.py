import requests
import json
from typing import Dict, Any, List, Optional

class InferenceClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def load_model(self, model_name: str, model_path: Optional[str] = None, 
                  precision: str = "fp16", device: str = "cuda") -> Dict[str, Any]:
        payload = {
            "model_name": model_name,
            "model_path": model_path,
            "precision": precision,
            "device": device
        }
        response = self.session.post(f"{self.base_url}/models/load", json=payload)
        response.raise_for_status()
        return response.json()
    
    def unload_model(self, model_name: str) -> Dict[str, Any]:
        response = self.session.post(f"{self.base_url}/models/unload", 
                                   json={"model_name": model_name})
        response.raise_for_status()
        return response.json()
    
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        response = self.session.post(f"{self.base_url}/models/switch", 
                                   json={"model_name": model_name})
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        response = self.session.get(f"{self.base_url}/models")
        response.raise_for_status()
        return response.json()
    
    def run_inference(self, model_name: str, input_data: Any, 
                     batch_size: int = 1, stream: bool = False) -> Dict[str, Any]:
        payload = {
            "model_name": model_name,
            "input": input_data,
            "batch_size": batch_size,
            "stream": stream
        }
        response = self.session.post(f"{self.base_url}/inference", json=payload)
        response.raise_for_status()
        return response.json()
    
    def get_system_status(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/system/status")
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/system/stats")
        response.raise_for_status()
        return response.json()