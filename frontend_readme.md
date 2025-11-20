# Quick Start

## interactive mode
python main.py

### Commands
Model managements
model load <name> [--model-path PATH] [--precision fp16|fp32|int8] [--device cuda|cpu]
model list
model unload <name>
model switch <name>

### Inference
infer --model <name> "<input text>" [--batch-size N] [--stream]
infer --model <name> --input-file <file_path> [--batch-size N] [--stream]

### System Monitoring
system status
system stats

### Interactive Commands
help                    # Show available commands
clear                   # Clear screen
exit, quit              # Exit CLI

### Input Examples(expected):
python main.py
> model load llama-2 --precision fp16 --device cuda
> infer --model llama-2 "Hello world"
> system status
> exit

### Ouput Examples(expected):
SUCCESS: Inference completed!
  Result: Response from model...
  Time: 0.325s  
