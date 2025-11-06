import requests
import os
from typing import Dict, Any, List, Optional
from pathlib import Path

class InferenceClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        env_base = os.getenv("APP_SERVER_URL")
        if env_base:
            base_url = env_base.strip()
        self.base_url = base_url
        self.session = requests.Session()

    # ----- models -----
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
        resp = self.session.get(f"{self.base_url}/models")
        resp.raise_for_status()
        data = resp.json()
        # Normalize to a list of model dicts (what the CLI expects)
        if isinstance(data, dict) and "models" in data:
            return data["models"]
        if isinstance(data, list):
            return data
        # Fallback to empty list if server responds unexpectedly
        return []

    # ----- inference -----
    def run_inference(self, model_name: str, input_data: Any,
                      batch_size: int = 1, stream: bool = False) -> Dict[str, Any]:
        params = {
            "model_name": model_name,
            "batch_size": batch_size,
            "stream": stream,
        }

        p = Path(str(input_data))
        if p.exists() and p.is_file():
            # multipart/form-data when a file path is provided
            with open(p, "rb") as f:
                files = {"input": f}
                response = self.session.post(
                    f"{self.base_url}/inference", data=params, files=files
                )
        else:
            # JSON body when input_data is text/JSON
            payload = dict(params)
            payload["input"] = input_data
            response = self.session.post(
                f"{self.base_url}/inference", json=payload
            )

        response.raise_for_status()
        return response.json()

    # ----- system -----
    def get_system_status(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/system/status")
        response.raise_for_status()
        return response.json()

    def get_statistics(self) -> Dict[str, Any]:
        response = self.session.get(f"{self.base_url}/system/stats")
        response.raise_for_status()
        return response.json()
