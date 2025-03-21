"""
XLA utilities for optimizing PyTorch code.

This module provides helper functions and utilities for writing PyTorch code that
is XLA-friendly and can be efficiently compiled by XLA when running on TPUs or
other accelerators that benefit from XLA compilation.

When XLA is not available, these utilities provide fallbacks that allow the same
code to run on standard PyTorch devices.
"""

import logging
import contextlib
import torch

# Set up logger
logger = logging.getLogger(__name__)

def is_xla_available():
    """Check if PyTorch XLA is available.
    
    Returns:
        bool: True if PyTorch XLA is available, False otherwise
    """
    try:
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False

def get_xla_supported_devices():
    """Get list of XLA-supported devices if available.
    
    Returns:
        list: List of supported XLA devices, or empty list if XLA not available
    """
    if not is_xla_available():
        return []
    
    try:
        import torch_xla.core.xla_model as xm
        return xm.get_xla_supported_devices()
    except (ImportError, AttributeError):
        return []

@contextlib.contextmanager
def xla_step():
    """Context manager for marking XLA computation steps.
    
    This is a no-op when XLA is not available, but marks step boundaries
    when using XLA. This helps XLA optimize computation across operations.
    """
    try:
        import torch_xla.core.xla_model as xm
        yield
        xm.mark_step()
    except ImportError:
        yield

def mark_step():
    """Mark an XLA step boundary.
    
    This is a no-op when XLA is not available, but marks step boundaries
    when using XLA. Call this after a batch of operations that should be
    compiled together.
    """
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step()
    except ImportError:
        pass

def get_xla_device(device_name=None):
    """Get an XLA device if available, otherwise fallback to standard device.
    
    Args:
        device_name (str, optional): Device name to use. Defaults to None.
    
    Returns:
        device: PyTorch device to use
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            if device_name in ["tpu", "xla"]:
                return xm.xla_device()
            elif device_name == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            else:
                return xm.xla_device()  # Default to first XLA device
        except ImportError:
            logger.warning("Failed to import torch_xla. Falling back to standard PyTorch devices.")
    
    # Fallback to standard devices
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def to_xla_tensor(tensor):
    """Convert a tensor to XLA format if XLA is available.
    
    Args:
        tensor (torch.Tensor): Tensor to convert
    
    Returns:
        torch.Tensor: Tensor on the appropriate device
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            return tensor.to(device)
        except ImportError:
            pass
    
    # Return original tensor if XLA not available
    return tensor

def optimizer_step(optimizer, barrier=True):
    """Perform an optimizer step, with XLA optimization if available.
    
    Args:
        optimizer (torch.optim.Optimizer): The optimizer
        barrier (bool, optional): Whether to use a barrier to synchronize 
                                 when using multiple XLA devices. Defaults to True.
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            optimizer.step()
            xm.mark_step()
        except ImportError:
            optimizer.step()
    else:
        optimizer.step()

def all_reduce(tensor, reduce_op="sum", groups=None):
    """All-reduce operation across XLA devices.
    
    This is a no-op when XLA is not available or when running on a single device.
    
    Args:
        tensor (torch.Tensor): Tensor to all-reduce
        reduce_op (str, optional): Reduction operation. Defaults to "sum".
        groups (list, optional): Process groups. Defaults to None.
    
    Returns:
        torch.Tensor: Reduced tensor
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            return xm.all_reduce(reduce_op, tensor, groups=groups)
        except ImportError:
            pass
    
    # Return original tensor if XLA not available
    return tensor

def save_to_tensor(tensor, hook=None):
    """Save a tensor for a backward hook in XLA.
    
    This is useful for custom backward operations in XLA. It's a no-op
    when XLA is not available.
    
    Args:
        tensor (torch.Tensor): Tensor to save
        hook (callable, optional): Hook to call during backward. Defaults to None.
    
    Returns:
        torch.Tensor: Original tensor
    """
    if is_xla_available():
        try:
            import torch_xla.core.xla_model as xm
            return xm.save(tensor, hook)
        except (ImportError, AttributeError):
            pass
    
    # Return original tensor if XLA not available
    return tensor 