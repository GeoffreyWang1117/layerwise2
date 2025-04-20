# -*- coding: utf-8 -*-
"""
Convert full PyTorch models to ONNX format

This utility converts complete PyTorch model files (.pth) to ONNX format.
It expects models saved with torch.save() containing the actual model object,
not just state dictionaries.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch.fx as fx

def ensure_directory(directory: str) -> None:
    """Ensure directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def is_complete_model(model_obj: Any) -> Tuple[bool, str, Optional[torch.nn.Module]]:
    """
    Check if the loaded object is a complete model or contains one.
    
    Args:
        model_obj: The loaded object from torch.load()
        
    Returns:
        Tuple of:
        - bool: True if a complete model was found
        - str: Message describing what was found
        - Optional[torch.nn.Module]: The model if found, None otherwise
    """
    # Case 1: Direct model
    if isinstance(model_obj, torch.nn.Module):
        return True, "Direct model object", model_obj
    
    # Case 2: Dictionary with 'model' key
    if isinstance(model_obj, dict) and 'model' in model_obj:
        if isinstance(model_obj['model'], torch.nn.Module):
            return True, "Model found in 'model' key", model_obj['model']
    
    # Case 3: Dictionary with any key containing a model
    if isinstance(model_obj, dict):
        for key, value in model_obj.items():
            if isinstance(value, torch.nn.Module):
                return True, f"Model found in '{key}' key", value
    
    # Not a complete model
    return False, "No complete model found", None

def convert_to_onnx(
    model_path: str, 
    output_path: str,
    input_shapes: Optional[Dict[str, Tuple]] = None,
    opset_version: int = 12
) -> Tuple[bool, str]:
    """
    Convert a complete PyTorch model to ONNX format.
    """
    print(f"\nLoading model from: {model_path}")
    
    try:
        # Load model using CPU to avoid CUDA issues
        loaded_obj = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Check if we have a complete model
        is_complete, message, model = is_complete_model(loaded_obj)
        
        if not is_complete or model is None:
            print(f"Error: {message}")
            print("The file doesn't contain a complete model that can be converted to ONNX.")
            print("Make sure you're using a model saved with the full architecture, not just a state dictionary.")
            return False, "No complete model found"
        
        print(f"Found model: {message}")
        print(f"Model type: {type(model).__name__}")
        
        # Set model to evaluation mode
        model.eval()
        
        # Extract any metadata if available
        metadata = {}
        if isinstance(loaded_obj, dict) and 'metadata' in loaded_obj:
            metadata = loaded_obj['metadata']
            print("Metadata found in model file")
        
        # Instead of using a specialized embedding-only approach, we now use torch.fx to trace the complete model and export it to ONNX
        print("Using torch.fx for complete model conversion")
        traced_model = fx.symbolic_trace(model)
        
        # Prepare dummy input using input_shapes if provided, otherwise default to a common shape
        if input_shapes is None:
            # You should adjust the default shape based on your model; here we use (1, 3, 224, 224) as an example
            print("No input shape provided. Using default shape (1, 3, 224, 224)")
            dummy_input = torch.randn(1, 3, 224, 224)
        else:
            # If input_shapes is provided as a dict, use the first key's shape
            if isinstance(input_shapes, dict):
                dummy_input = torch.randn(*list(input_shapes.values())[0])
            else:
                dummy_input = torch.randn(*input_shapes)
        
        print("Exporting ONNX using the traced complete model...")
        torch.onnx.export(
            traced_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            verbose=True
        )
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"\nSuccess! ONNX model exported to: {output_path} ({file_size:.2f} MB)")
            try:
                import onnx
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                print("ONNX model verified successfully")
            except ImportError:
                print("Install the 'onnx' package to verify the model")
            except Exception as e:
                print(f"Warning: ONNX model verification failed: {e}")
            return True, "Conversion successful"
        else:
            return False, "ONNX file not created"
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False, f"Error during conversion: {str(e)}"

def main():
    """Main function handling command line arguments."""
    # Create argument parser with descriptive help
    parser = argparse.ArgumentParser(
        description="Convert PyTorch models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a model with default settings:
  python utils/pth2onnx0.py --model_name teacher_frappe_full

  # Specify input shapes (for image models):
  python utils/pth2onnx0.py --model_name my_model --input_shape 1 3 224 224

  # Specify multiple named inputs (for multi-input models):
  python utils/pth2onnx0.py --model_name complex_model --named_input user 1 1 --named_input item 1 1
"""
    )
    
    # Add arguments
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model file in outputs folder (without .pth extension)")
    parser.add_argument("--output_name", type=str, default=None,
                        help="Name for the output ONNX file (without extension)")
    parser.add_argument("--input_shape", type=int, nargs='+',
                        help="Shape of the input tensor (e.g., 1 3 224 224 for a batch of one RGB image)")
    parser.add_argument("--named_input", type=str, nargs='+', action='append',
                        help="Named input with shape (e.g., --named_input input_name 1 5 10)")
    parser.add_argument("--opset_version", type=int, default=12,
                        help="ONNX opset version (default: 12)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Ensure output directory exists
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    ensure_directory(outputs_dir)
    
    # Build the input model path
    model_name = args.model_name
    if not model_name.endswith('.pth'):
        model_name += '.pth'
    model_path = os.path.join(outputs_dir, model_name)
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        print("Make sure the file exists in the outputs directory.")
        sys.exit(1)
    
    # Build the output path
    if args.output_name:
        output_name = args.output_name
    else:
        output_name = args.model_name.replace('.pth', '')
    
    if not output_name.endswith('.onnx'):
        output_name += '.onnx'
    output_path = os.path.join(outputs_dir, output_name)
    
    # Process input shapes
    input_shapes = None
    
    # Handle single input shape
    if args.input_shape:
        input_shapes = {'input': tuple(args.input_shape)}
    
    # Handle named inputs (overrides input_shape if both provided)
    if args.named_input:
        input_shapes = {}
        for named_input in args.named_input:
            name = named_input[0]
            shape = tuple(int(x) for x in named_input[1:])
            input_shapes[name] = shape
    
    # Convert the model
    success, message = convert_to_onnx(
        model_path=model_path,
        output_path=output_path,
        input_shapes=input_shapes,
        opset_version=args.opset_version
    )
    
    if not success:
        print(f"\nConversion failed: {message}")
        sys.exit(1)
    else:
        print(f"\nConversion completed successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()