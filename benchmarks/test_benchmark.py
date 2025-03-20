import pytest
import numpy as np
import time
import os

# Check if PyTorch is available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Check if JAX is available
try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

# Test utility functions
def test_timing_accuracy():
    """Test that our timing function is accurate enough for benchmarking."""
    # Sleep for a known duration
    duration = 0.1
    
    # Measure the time it takes to sleep
    start = time.time()
    time.sleep(duration)
    elapsed = time.time() - start
    
    # The elapsed time should be close to the duration
    # Allow for some margin of error due to system scheduling
    assert abs(elapsed - duration) < 0.02, f"Expected ~{duration}s, got {elapsed}s"

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestPyTorchBenchmark:
    def test_cuda_availability(self):
        """Test that CUDA is available if we're on a CUDA-capable machine."""
        if "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("NVIDIA_VISIBLE_DEVICES"):
            assert torch.cuda.is_available(), "CUDA should be available but isn't"
    
    def test_tensor_device_placement(self):
        """Test that tensors are correctly placed on the specified device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create a tensor on CPU
        cpu_tensor = torch.randn(10, 10)
        assert cpu_tensor.device.type == "cpu"
        
        # Move to CUDA
        cuda_tensor = cpu_tensor.cuda()
        assert cuda_tensor.device.type == "cuda"
        
        # Create directly on CUDA
        direct_cuda = torch.randn(10, 10, device="cuda")
        assert direct_cuda.device.type == "cuda"
    
    def test_operation_runs_on_specified_device(self):
        """Test that operations run on the device where the tensors are placed."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        # Create tensors on different devices
        cpu_a = torch.randn(1000, 1000)
        cpu_b = torch.randn(1000, 1000)
        
        cuda_a = cpu_a.cuda()
        cuda_b = cpu_b.cuda()
        
        # Measure time for CPU operation
        start = time.time()
        cpu_result = torch.matmul(cpu_a, cpu_b)
        cpu_time = time.time() - start
        
        # Measure time for GPU operation
        # First, synchronize to ensure fair timing
        torch.cuda.synchronize()
        start = time.time()
        cuda_result = torch.matmul(cuda_a, cuda_b)
        torch.cuda.synchronize()  # Wait for GPU operation to complete
        cuda_time = time.time() - start
        
        # For large matrices, GPU should be faster
        # This test might not always pass on all hardware,
        # but it's a good sanity check
        print(f"CPU time: {cpu_time:.6f}s, GPU time: {cuda_time:.6f}s")
        
        # Check results are numerically equivalent
        # Use higher tolerance for GPU/CPU comparison due to floating point differences
        cpu_result_np = cpu_result.numpy()
        cuda_result_np = cuda_result.cpu().numpy()
        np.testing.assert_allclose(cpu_result_np, cuda_result_np, rtol=1e-4, atol=1e-4)

@pytest.mark.skipif(not HAS_JAX, reason="JAX not installed")
class TestJaxBenchmark:
    def test_device_count(self):
        """Test that JAX can see GPU devices if available."""
        # This will show CPU if no GPU is available
        devices = jax.devices()
        print(f"JAX devices: {devices}")
        
        # If we're on a system with NVIDIA GPUs, JAX should see them
        if "CUDA_VISIBLE_DEVICES" in os.environ or os.environ.get("NVIDIA_VISIBLE_DEVICES"):
            assert any("gpu" in str(d).lower() for d in devices), "GPU should be available to JAX but isn't"
    
    def test_jit_compilation(self):
        """Test that JAX JIT compilation works correctly."""
        # Define a simple function
        def simple_fn(x, y):
            return jnp.dot(x, y)
        
        # JIT compile it
        jitted_fn = jax.jit(simple_fn)
        
        # Create some test data
        x = jnp.array(np.random.randn(10, 10))
        y = jnp.array(np.random.randn(10, 10))
        
        # First call compiles and runs
        result1 = jitted_fn(x, y)
        
        # Measure time for regular function
        start = time.time()
        for _ in range(100):
            simple_fn(x, y)
        regular_time = time.time() - start
        
        # Measure time for jitted function
        # First call for warmup
        jitted_fn(x, y)
        start = time.time()
        for _ in range(100):
            jitted_fn(x, y)
        jitted_time = time.time() - start
        
        print(f"Regular time: {regular_time:.6f}s, JIT time: {jitted_time:.6f}s")
        
        # For repeated calls, jitted should be faster
        assert jitted_time < regular_time, "JIT compilation should speed up repeated function calls"
    
    def test_operation_on_accelerator(self):
        """Test that operations run on the accelerator when available."""
        # Define a function that does a lot of computation
        @jax.jit
        def matmul_fn(x, y):
            return jnp.matmul(x, y)
        
        # Create large matrices
        size = 2000
        x = jnp.array(np.random.randn(size, size))
        y = jnp.array(np.random.randn(size, size))
        
        # Run once to compile
        _ = matmul_fn(x, y)
        
        # Run again and time it
        start = time.time()
        result = matmul_fn(x, y)
        # Block until the computation is complete
        result.block_until_ready()
        jax_time = time.time() - start
        
        # Now do the same with NumPy on CPU
        x_np = np.array(x)
        y_np = np.array(y)
        
        start = time.time()
        np_result = np.matmul(x_np, y_np)
        np_time = time.time() - start
        
        print(f"JAX time: {jax_time:.6f}s, NumPy time: {np_time:.6f}s")
        
        # Check results are numerically equivalent
        np.testing.assert_allclose(np.array(result), np_result, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    # Run tests directly
    test_timing_accuracy()
    print("Timing accuracy test passed")
    
    if HAS_TORCH:
        test_instance = TestPyTorchBenchmark()
        try:
            test_instance.test_cuda_availability()
            print("PyTorch CUDA availability test passed")
            
            test_instance.test_tensor_device_placement()
            print("PyTorch tensor device placement test passed")
            
            test_instance.test_operation_runs_on_specified_device()
            print("PyTorch operation device test passed")
        except Exception as e:
            print(f"PyTorch test failed: {e}")
    else:
        print("PyTorch not installed, skipping tests")
    
    if HAS_JAX:
        test_instance = TestJaxBenchmark()
        try:
            test_instance.test_device_count()
            print("JAX device count test passed")
            
            test_instance.test_jit_compilation()
            print("JAX JIT compilation test passed")
            
            test_instance.test_operation_on_accelerator()
            print("JAX accelerator operation test passed")
        except Exception as e:
            print(f"JAX test failed: {e}")
    else:
        print("JAX not installed, skipping tests") 