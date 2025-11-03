from api.client import InferenceClient

class InferenceCommands:  # ← Make sure it's exactly this name
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def run_inference(self, model_name: str, input_data: str, 
                     batch_size: int = 1, stream: bool = False):
        print(f"RUNNING: Running inference with model: {model_name}")
        print(f"  Input: {input_data}")
        
        try:
            result = self.client.run_inference(
                model_name=model_name,
                input_data=input_data,  # ← Note: this should be 'input_data'
                batch_size=batch_size,
                stream=stream
            )
            
            print(f"SUCCESS: Inference completed!")
            print(f"  Result: {result.get('output', 'No output')}")
            print(f"  Processing Time: {result.get('processing_time', 'N/A')}s")
            
        except Exception as e:
            print(f"ERROR: Error running inference: {e}")