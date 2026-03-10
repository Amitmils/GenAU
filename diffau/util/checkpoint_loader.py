import pickle
import torch
import os
from pathlib import Path, PurePath
import sys


def load_checkpoint_cross_platform(checkpoint_path, **kwargs):
    """
    Load a PyTorch Lightning checkpoint with cross-platform path compatibility.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        **kwargs: Additional arguments to pass to the model loading
        
    Returns:
        The loaded checkpoint dictionary
    """
    # Apply the compatibility fix before loading    
    try:
        # Load the raw checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return checkpoint
    except Exception as e:
        if "PosixPath" in str(e):
            print(f"Attempting to fix PosixPath compatibility issue...")
            
            # Alternative approach: manually fix the checkpoint file
            try:
                return _load_with_path_conversion(checkpoint_path)
            except Exception as e2:
                raise RuntimeError(f"Failed to load checkpoint due to path compatibility issues. "
                                 f"Original error: {e}. Alternative approach error: {e2}")
        else:
            raise e


def _load_with_path_conversion(checkpoint_path):
    """
    Alternative loading method that converts path objects to strings.
    """
    import pickle
    
    class PathPickleLoader:
        def __init__(self):
            self.original_loads = pickle.loads
            
        def __enter__(self):
            pickle.loads = self._patched_loads
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pickle.loads = self.original_loads
            
        def _patched_loads(self, data):
            # Custom unpickler that handles path objects
            import io
            
            class PathUnpickler(pickle.Unpickler):
                def load_build(self):
                    stack = self.stack
                    state = stack.pop()
                    inst = stack[-1]
                    
                    # Convert any path objects to strings
                    if hasattr(inst, '__dict__'):
                        for key, value in inst.__dict__.items():
                            if isinstance(value, PurePath):
                                setattr(inst, key, str(value))
                    
                    # Handle state dictionary
                    if isinstance(state, dict):
                        for key, value in state.items():
                            if isinstance(value, PurePath):
                                state[key] = str(value)
                    
                    # Call original load_build
                    setattr(inst, '__dict__', state) if hasattr(inst, '__dict__') else None
                    
            return PathUnpickler(io.BytesIO(data)).load()
    
    # Load with path conversion
    with PathPickleLoader():
        return torch.load(checkpoint_path, map_location='cpu')


def safe_load_from_checkpoint(model_cls, checkpoint_path, **kwargs):
    """
    Safely load a PyTorch Lightning model from checkpoint with cross-platform compatibility.
    
    Args:
        model_cls: The model class (e.g., ScoreModel)
        checkpoint_path: Path to the checkpoint file
        **kwargs: Additional arguments to pass to the model
        
    Returns:
        The loaded model instance
    """
    # Apply compatibility fixes    
    try:
        # Try the standard loading method first
        return model_cls.load_from_checkpoint(checkpoint_path, **kwargs)
    except Exception as e:
        if "PosixPath" in str(e) or "cannot instantiate" in str(e):
            print(f"Standard loading failed due to path compatibility. Attempting alternative method...")
            
            # Load checkpoint manually and create model
            checkpoint = load_checkpoint_cross_platform(checkpoint_path)
            
            # Extract hyperparameters
            if 'hyper_parameters' in checkpoint:
                hparams = checkpoint['hyper_parameters']
                # Merge with any additional kwargs
                hparams.update(kwargs)
            else:
                hparams = kwargs
            
            # Create model instance
            model = model_cls(**hparams)
            
            # Load state dict
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            return model
        else:
            raise e
