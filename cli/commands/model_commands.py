from api.client import InferenceClient
import time

class ModelCommands:
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def load_model(self, model_name: str, model_path: str = None, 
                  precision: str = "fp16", device: str = "cuda"):
        print(f"Loading model: {model_name}")
        start_time = time.time()
        try:
            result = self.client.load_model(
                model_name=model_name,
                model_path=model_path,
                precision=precision,
                device=device
            )
            end_time = time.time()
            load_time = end_time - start_time
            print(f"SUCCESS: Model loaded successfully!")
            print(f"  Name: {result.get('model_name')}")
            print(f"  Status: {result.get('status')}")
            print(f"  Time: {load_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            load_time = end_time - start_time
            print(f"ERROR: Error loading model: {e}")
            print(f"  Time elapsed: {load_time:.2f}s")
    
    def list_models(self):
        start_time = time.time()
        try:
            models = self.client.list_models()
            end_time = time.time()
            query_time = end_time - start_time
            print("LOADED MODELS:")
            for model in models:
                status = "[LOADED]" if model.get('loaded') else "[NOT LOADED]"
                print(f"  {status} {model.get('name', 'Unknown')}")
            print(f"Time: {query_time:.3f}s")
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            print(f"ERROR: Error listing models: {e}")
            print(f"  Time elapsed: {query_time:.2f}s")
    
    def unload_model(self, model_name: str):
        start_time = time.time()
        try:
            result = self.client.unload_model(model_name)
            end_time = time.time()
            unload_time = end_time - start_time
            print(f"SUCCESS: Model {model_name} unloaded successfully!")
            print(f"  Time: {unload_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            unload_time = end_time - start_time
            print(f"ERROR: Error unloading model: {e}")
            print(f"  Time elapsed: {unload_time:.2f}s")
    
    def switch_model(self, model_name: str):
        start_time = time.time()
        try:
            result = self.client.switch_model(model_name)
            end_time = time.time()
            switch_time = end_time - start_time
            print(f"SUCCESS: Switched to model: {model_name}")
            print(f"  Time: {switch_time:.2f}s")
        except Exception as e:
            end_time = time.time()
            switch_time = end_time - start_time
            print(f"ERROR: Error switching model: {e}")
            print(f"  Time elapsed: {switch_time:.2f}s")