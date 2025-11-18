from api.client import InferenceClient
import time

class InferenceCommands:
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def run_inference(self, model_name: str, input_data: str, 
                     batch_size: int = 1, stream: bool = False):
        print(f"RUNNING: Running inference with model: {model_name}")
        print(f"  Input: {input_data}")
        
        start_time = time.time()
        try:
            result = self.client.run_inference(
                model_name=model_name,
                input=input_data,
                batch_size=batch_size,
                stream=stream
            )
            end_time = time.time()
            inference_time = end_time - start_time
            
            print(f"SUCCESS: Inference completed!")
            print(f"  Result: {result.get('output', 'No output')}")
            print(f"  Time: {inference_time:.3f}s")
            
        except Exception as e:
            end_time = time.time()
            inference_time = end_time - start_time
            print(f"ERROR: Error running inference: {e}")
            print(f"  Time elapsed: {inference_time:.2f}s")