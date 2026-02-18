# alspec-implementation

This repository contains a GPU implementation of attention level speculation to reduce LLM inference latency. Additional sections include a testbench, demo interface, and backend server.

## Running Baseline

~/implementation/server-core/llama.cpp/build/bin

### No Flash Attn

./llama-cli --flash_attn off -m ~/implementation/server-core/model_storage/Llama-3.2-1B-Instruct-Q4_K_M.gguf --system-prompt "you are a helpful assistant." --prompt "Where is the eifel tower located" -n 256 2>"/home/alspec/implementation/server-core/logs/log_$(date +%Y%m%d_%H%M%S)_${MODEL}.txt"


### Yes Flash Attn
