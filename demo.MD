# DEMO

## Inference Engine

verify that pwd = ~/implementation/server-core/llama.cpp/build/bin

./llama-cli -m ~/implementation/server-core/model_storage/Llama-3.2-1B-Instruct-Q4_K_M.gguf --system-prompt "you are a helpful assistant." -n 256 2>"/home/alspec/implementation/server-core/logs/log_$(date +%Y%m%d_%H%M%S)_${MODEL}.txt"

## Websocket

### Terminal 1:

pwd = ~/implementation

source /home/alspec/implementation/serverVenv/bin/activate
uvicorn api.server:app --host 0.0.0.0 --port 8000

### Terminal 2:

pwd = ~/implementation/

source /home/alspec/implementation/serverVenv/bin/activate
export APP_SERVER_URL="http://localhost:8000"
python3 main.py chat

## CLI

pwd = ~/implementation/

source /home/alspec/implementation/serverVenv/bin/activate
python3 main.py