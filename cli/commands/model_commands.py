from api.client import InferenceClient

class ModelCommands:
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def load_model(self, model_name: str, model_path: str = None, 
                  precision: str = "fp16", device: str = "cuda"):
        print(f"Loading model: {model_name}")
        try:
            result = self.client.load_model(
                model_name=model_name,
                model_path=model_path,
                precision=precision,
                device=device
            )
            print(f"SUCCESS: Model loaded successfully!")
            print(f"  Name: {result.get('model_name')}")
            print(f"  Status: {result.get('status')}")
        except Exception as e:
            print(f"ERROR: Error loading model: {e}")
    
    def list_models(self):
        try:
            models = self.client.list_models()
            print("LOADED MODELS:")
            for model in models:
                status = "[LOADED]" if model.get('loaded') else "[NOT LOADED]"
                print(f"  {status} {model.get('name', 'Unknown')}")
        except Exception as e:
            print(f"ERROR: Error listing models: {e}")
    
    def unload_model(self, model_name: str):
        try:
            result = self.client.unload_model(model_name)
            print(f"SUCCESS: Model {model_name} unloaded successfully!")
        except Exception as e:
            print(f"ERROR: Error unloading model: {e}")
    
    def switch_model(self, model_name: str):
        try:
            result = self.client.switch_model(model_name)
            print(f"SUCCESS: Switched to model: {model_name}")
        except Exception as e:
            print(f"ERROR: Error switching model: {e}")