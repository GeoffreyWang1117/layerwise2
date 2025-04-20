# filepath: /home/coder-gw/layerwise2/utils/pth2onnx.py

"""
Simple PyTorch model (.pth) to ONNX format converter
"""

import os
import sys
import torch  # Make sure this import is correct
import argparse
from pathlib import Path

def convert_model(model_path, output_path):
    """Convert a PyTorch model to ONNX format."""
    print(f"Loading model from: {model_path}")
    
    try:
        # Handle CUDA models
        device = torch.device("cpu")
        loaded_obj = torch.load(model_path, map_location=device)
        
        # Check if model or state dict
        if isinstance(loaded_obj, torch.nn.Module):
            model = loaded_obj
            print("Loaded a complete model")
            
            # Set model to eval mode
            model.eval()
            
            # Simple dummy input - adjust as needed
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Export to ONNX
            print(f"Exporting to ONNX: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            )
            
            if os.path.exists(output_path):
                print(f"ONNX file created: {output_path}")
                return True
            else:
                print("Failed to create ONNX file")
                return False
                
        else:
            print("Loaded a state dictionary, not a complete model.")
            print("To convert to ONNX, you need to define the model architecture first.")
            return False
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch models to ONNX format")
    parser.add_argument("--model_path", required=True, help="Path to .pth model")
    parser.add_argument("--output_dir", default="outputs", help="Output directory")
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set output filename
    base_name = os.path.basename(args.model_path).replace('.pth', '')
    output_path = os.path.join(args.output_dir, f"{base_name}.onnx")
    
    # Convert the model
    success = convert_model(args.model_path, output_path)
    
    if success:
        print("Conversion successful")
        sys.exit(0)
    else:
        print("Conversion failed")
        sys.exit(1)

if __name__ == "__main__":
    main()