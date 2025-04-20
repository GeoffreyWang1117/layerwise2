import os
import sys
import torch
import argparse
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from pathlib import Path
import traceback

def create_dummy_input(model_info, device='cuda'):
    """
    Create dummy inputs for the model based on model information.
    
    Args:
        model_info: Dictionary containing model metadata
        device: Device to create tensors on
    
    Returns:
        Dictionary of dummy inputs
    """
    dummy_inputs = {}
    
    # Check if we have sparse features information
    if 'sparse_features' in model_info:
        for feat_name in model_info['sparse_features']:
            # Create dummy input tensor with batch size 2
            dummy_inputs[feat_name] = torch.tensor([[0], [1]], dtype=torch.long, device=device)
    else:
        # Default dummy input if no feature info available
        dummy_inputs = torch.randn(2, 10, device=device)
    
    return dummy_inputs

def visualize_model_architecture(model):
    """
    Generate a text-based visualization of model architecture
    
    Args:
        model: PyTorch model
        
    Returns:
        String representation of model architecture
    """
    layers_info = []
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        layers_info.append(f"{name}: {list(param.shape)} ({param_count} parameters)")
    
    header = f"Model Architecture: {model.__class__.__name__}"
    separator = "=" * len(header)
    summary = f"{header}\n{separator}\n"
    summary += "\n".join(layers_info)
    summary += f"\n\nTotal parameters: {total_params:,}"
    summary += f"\nTrainable parameters: {trainable_params:,}"
    
    return summary

def visualize_model(model_path, output_dir=None, max_depth=None):
    """
    Visualize a PyTorch model using TensorBoard.
    
    Args:
        model_path: Path to the model file
        output_dir: Directory to save TensorBoard logs
        max_depth: Maximum depth for visualization
    """
    print(f"Loading model from: {model_path}")
    
    try:
        # Load the model to CUDA
        checkpoint = torch.load(model_path, map_location=torch.device('cuda'))
        
        # Determine the model format
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Full model package
            model = checkpoint['model']
            model_info = checkpoint.get('metadata', {})
        elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # State dict only - we can't visualize this directly
            print("Error: Found only state dictionary, not a complete model.")
            print("Please provide a full model file that includes the model architecture.")
            return
        else:
            # Assume it's a direct model
            model = checkpoint
            model_info = {}
        
        # Ensure model is on CUDA and in evaluation mode
        model = model.cuda()
        model.eval()
        
        # Look for model info file if metadata is empty
        if not model_info and isinstance(checkpoint, dict):
            info_path = model_path.replace('.pth', '_info.json')
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)
                print(f"Loaded additional model info from: {info_path}")
        
        # Create dummy input on CUDA
        dummy_input = create_dummy_input(model_info, device='cuda')
        
        # Setup TensorBoard writer
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(model_path)), 'tensorboard_logs')
        
        os.makedirs(output_dir, exist_ok=True)
        model_name = os.path.basename(model_path).split('.')[0]
        writer = SummaryWriter(log_dir=os.path.join(output_dir, model_name))
        
        # Generate text-based model architecture visualization
        architecture_text = visualize_model_architecture(model)
        print("\nModel Architecture Summary:")
        print(architecture_text)
        
        # Add model architecture as text to TensorBoard
        writer.add_text("Model Architecture", architecture_text.replace("\n", "  \n"), global_step=0)
        
        # Try to run a forward pass and record the output
        print("\nAttempting to run model inference...")
        try:
            with torch.no_grad():
                if isinstance(dummy_input, dict):
                    output = model(dummy_input)
                else:
                    output = model(dummy_input)
                
                # Record input and output tensors
                if isinstance(dummy_input, dict):
                    for name, tensor in dummy_input.items():
                        writer.add_histogram(f"inputs/{name}", tensor, global_step=0)
                else:
                    writer.add_histogram("inputs/input", dummy_input, global_step=0)
                    
                if isinstance(output, dict):
                    for name, tensor in output.items():
                        writer.add_histogram(f"outputs/{name}", tensor, global_step=0)
                else:
                    writer.add_histogram("outputs/output", output, global_step=0)
                
                print(f"Inference successful - output shape: {output.shape if hasattr(output, 'shape') else 'N/A'}")
        except Exception as e:
            print(f"Error running model inference: {e}")
            traceback.print_exc()
            print("Continuing with parameter visualization...")
        
        # Add model parameters as histograms
        print("\nAdding parameter histograms to TensorBoard...")
        for name, param in model.named_parameters():
            writer.add_histogram(f"parameters/{name}", param, global_step=0)
            
            # If gradient information is available, add it too
            if param.grad is not None:
                writer.add_histogram(f"gradients/{name}", param.grad, global_step=0)
        
        writer.close()
        
        print(f"\nVisualization complete. Run 'tensorboard --logdir={output_dir}' to view.")
        print(f"Then open http://localhost:6006 in your browser.")
        
    except Exception as e:
        print(f"Error visualizing model: {e}")
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Visualize PyTorch models using TensorBoard")
    parser.add_argument("model_name", type=str, help="Name of model file in outputs directory (with or without .pth extension)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save TensorBoard logs")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth for visualization")
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("Error: CUDA is not available on this system.")
        print("This script requires CUDA for model visualization.")
        return
    
    # Get the outputs directory
    outputs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
    
    # Find the model file
    model_name = args.model_name
    if not model_name.endswith('.pth'):
        model_name += '.pth'
    
    model_path = os.path.join(outputs_dir, model_name)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        
        # List available models
        print("\nAvailable models:")
        for file in os.listdir(outputs_dir):
            if file.endswith('.pth'):
                print(f"  - {file}")
        return
    
    print(f"Using CUDA for model visualization: {torch.cuda.get_device_name(0)}")
    
    # Visualize the model
    visualize_model(model_path, args.output_dir, args.max_depth)

if __name__ == "__main__":
    main()