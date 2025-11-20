import argparse
import sys
import shlex
import asyncio
from api.client import InferenceClient
from cli.commands.model_commands import ModelCommands
from cli.commands.inference_commands import InferenceCommands
from cli.commands.system_commands import SystemCommands
from cli.chat import chat_session

class InferenceCLI:
    def __init__(self):
        self.client = InferenceClient()
        self.model_commands = ModelCommands(self.client)
        self.inference_commands = InferenceCommands(self.client)
        self.system_commands = SystemCommands(self.client)
        self.parser = self.setup_parser()
        
    def setup_parser(self):
        parser = argparse.ArgumentParser(description='Distributed Inference CLI')
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Model management commands
        model_parser = subparsers.add_parser('model', help='Model operations')
        model_subparsers = model_parser.add_subparsers(dest='model_command')
        
        load_parser = model_subparsers.add_parser('load', help='Load a model')
        load_parser.add_argument('model_name', help='Name of the model to load')
        load_parser.add_argument('--model-path', help='Path to model files')
        load_parser.add_argument('--precision', choices=['fp16', 'fp32', 'int8'], default='fp16')
        load_parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
        
        model_subparsers.add_parser('list', help='List available models')
        
        unload_parser = model_subparsers.add_parser('unload', help='Unload a model')
        unload_parser.add_argument('model_name', help='Name of the model to unload')
        
        switch_parser = model_subparsers.add_parser('switch', help='Switch active model')
        switch_parser.add_argument('model_name', help='Name of the model to switch to')
        
        # Inference commands
        inference_parser = subparsers.add_parser('infer', help='Run inference')
        inference_parser.add_argument('--model', required=True, help='Model to use for inference')
        inference_parser.add_argument('--input', required=True, help='Input data or file path')
        inference_parser.add_argument('--batch-size', type=int, default=1)
        inference_parser.add_argument('--output', help='Output file path')
        inference_parser.add_argument('--stream', action='store_true', help='Stream results')
        
        # System commands
        system_parser = subparsers.add_parser('system', help='System operations')
        system_subparsers = system_parser.add_subparsers(dest='system_command')
        system_subparsers.add_parser('status', help='Check system status')
        system_subparsers.add_parser('stats', help='Show performance statistics')
        
        # Chat command (WebSocket)
        subparsers.add_parser('chat', help='Interactive chat session over WebSocket')
        
        return parser
    
    def handle_model_command(self, args):
        if args.model_command == 'load':
            self.model_commands.load_model(
                args.model_name, 
                args.model_path, 
                args.precision, 
                args.device
            )
        elif args.model_command == 'list':
            self.model_commands.list_models()
        elif args.model_command == 'unload':
            self.model_commands.unload_model(args.model_name)
        elif args.model_command == 'switch':
            self.model_commands.switch_model(args.model_name)
    
    def handle_inference_command(self, args):
        self.inference_commands.run_inference(
            args.model,
            args.input,
            args.batch_size,
            args.stream
        )
    
    def handle_system_command(self, args):
        if args.system_command == 'status':
            self.system_commands.get_status()
        elif args.system_command == 'stats':
            self.system_commands.get_statistics()
    
    def run_command(self, command_string):
        """Execute a command string"""
        try:
            # Parse the command
            args = self.parser.parse_args(shlex.split(command_string))
            
            if args.command == 'model':
                self.handle_model_command(args)
            elif args.command == 'infer':
                self.handle_inference_command(args)
            elif args.command == 'system':
                self.handle_system_command(args)
            elif args.command == 'chat':
                asyncio.run(chat_session())            
                
        except SystemExit:
            # argparse calls sys.exit() on help or error, we want to continue
            pass
        except Exception as e:
            print(f"Error: {e}")
    
    def run_interactive(self):
        """Run in interactive loop mode"""
        print("Distributed Inference CLI - Interactive Mode")
        print("Type commands below (or 'exit' to quit)")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("> ").strip()
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit']:
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not user_input:
                    continue
                
                # Execute the command
                self.run_command(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break

def main():
    cli = InferenceCLI()
    
    # If no command line arguments, run in interactive mode
    if len(sys.argv) == 1:
        cli.run_interactive()
    else:
        # Original single-command behavior
        args = cli.parser.parse_args()
        
        if not args.command:
            cli.parser.print_help()
            return
        
        try:
            if args.command == 'model':
                cli.handle_model_command(args)
            elif args.command == 'infer':
                cli.handle_inference_command(args)
            elif args.command == 'system':
                cli.handle_system_command(args)
            elif args.command == 'chat':
                asyncio.run(chat_session())
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()