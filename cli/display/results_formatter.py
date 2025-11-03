# cli/display/results_formatter.py
from typing import Dict, Any, List
import json

class ResultsFormatter:
    @staticmethod
    def display_model_loaded(result: Dict[str, Any]):
        print(f"✅ Model loaded successfully!")
        print(f"   Name: {result.get('model_name')}")
        print(f"   Status: {result.get('status')}")
        print(f"   Load Time: {result.get('load_time', 'N/A')}s")
    
    @staticmethod
    def display_model_list(models: List[Dict[str, Any]]):
        if not models:
            print("No models loaded")
            return
        
        print("\n📚 Loaded Models:")
        print("-" * 50)
        for model in models:
            status = "🟢" if model.get('loaded') else "🔴"
            print(f"{status} {model.get('name', 'Unknown'):<20} "
                  f"| Device: {model.get('device', 'N/A'):<8} "
                  f"| Precision: {model.get('precision', 'N/A'):<6}")
    
    @staticmethod
    def display_inference_result(result: Dict[str, Any]):
        print(f"\n🎯 Inference Results:")
        print("-" * 30)
        print(f"Model: {result.get('model_name')}")
        print(f"Processing Time: {result.get('processing_time', 'N/A')}s")
        print(f"Output Shape: {result.get('output_shape', 'N/A')}")
        
        # Display output data (truncated if too long)
        output = result.get('output', 'No output')
        if isinstance(output, str) and len(output) > 200:
            output = output[:200] + "..."
        print(f"Output: {output}")