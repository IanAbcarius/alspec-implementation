from api.client import InferenceClient

class SystemCommands:
    def __init__(self, client: InferenceClient):
        self.client = client
    
    def get_status(self):
        try:
            status = self.client.get_system_status()
            print("🖥️  System Status:")
            print(f"   Server: {status.get('server_status', 'Unknown')}")
            print(f"   GPU: {status.get('gpu_status', 'Unknown')}")
            print(f"   Active Model: {status.get('active_model', 'None')}")
            print(f"   Memory Usage: {status.get('memory_usage', 'N/A')}")
        except Exception as e:
            print(f"❌ Error getting system status: {e}")
    
    def get_statistics(self):
        try:
            stats = self.client.get_statistics()
            print("📊 System Statistics:")
            print(f"   Total Inferences: {stats.get('total_inferences', 0)}")
            print(f"   Average Latency: {stats.get('avg_latency', 'N/A')}ms")
            print(f"   Models Loaded: {stats.get('models_loaded', 0)}")
        except Exception as e:
            print(f"❌ Error getting statistics: {e}")