# api/smoke_test.py
import json
from pathlib import Path
from api.client import InferenceClient

c = InferenceClient(base_url="http://127.0.0.1:8000")

print("==> list_models")
print(json.dumps(c.list_models(), indent=2))

print("==> load_model demo-new")
print(json.dumps(c.load_model("demo-new", precision="fp16", device="cuda"), indent=2))

print("==> switch_model demo-base")
print(json.dumps(c.switch_model("demo-base"), indent=2))

print("==> system status")
print(json.dumps(c.get_system_status(), indent=2))

Path("sample.txt").write_text("hello from client")
print("==> run_inference with file")
print(json.dumps(c.run_inference("demo-base", "sample.txt"), indent=2))

print("==> system stats")
print(json.dumps(c.get_statistics(), indent=2))
