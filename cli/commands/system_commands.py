from api.client import InferenceClient
import time

class SystemCommands:
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def get_status(self):
        start_time = time.time()
        try:
            status = self.client.get_system_status()
            end_time = time.time()
            query_time = end_time - start_time
            print("SYSTEM STATUS:")
            print(f"  Server: {status.get('server_status', 'Unknown')}")
            print(f"  GPU: {status.get('gpu_status', 'Unknown')}")
            print(f"  Active Model: {status.get('active_model', 'None')}")
            print(f"  Memory Usage: {status.get('memory_usage', 'N/A')}")
            print(f"Time: {query_time:.3f}s")
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            print(f"ERROR: Error getting system status: {e}")
            print(f"  Time elapsed: {query_time:.2f}s")
    
    def get_statistics(self):
        start_time = time.time()
        try:
            stats = self.client.get_statistics()
            end_time = time.time()
            query_time = end_time - start_time
            print("SYSTEM STATISTICS:")
            print(f"  Total Inferences: {stats.get('total_inferences', 0)}")
            print(f"  Average Latency: {stats.get('avg_latency', 'N/A')}ms")
            print(f"  Models Loaded: {stats.get('models_loaded', 0)}")
            print(f"Time: {query_time:.3f}s")
        except Exception as e:
            end_time = time.time()
            query_time = end_time - start_time
            print(f"ERROR: Error getting statistics: {e}")
            print(f"  Time elapsed: {query_time:.2f}s")