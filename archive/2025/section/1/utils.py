import torch
import matplotlib.pyplot as plt
import time
import numpy as np

def plot_matrix(matrix, title=""):
    """Plot a matrix with a colorbar."""
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.numpy())
    plt.title(title)
    plt.colorbar()
    plt.show()

def plot_singular_values(S, title="Singular Values"):
    """Plot singular values on a log scale."""
    plt.figure(figsize=(10, 4))
    plt.semilogy(S.numpy(), 'bo-')
    plt.title(title)
    plt.grid(True)
    plt.xlabel("Index")
    plt.ylabel("Singular Value (log scale)")
    plt.show()

def time_access(A, access_type="row", n_trials=1000):
    """Time row vs column access patterns."""
    start = time.time()
    if access_type == "row":
        for _ in range(n_trials):
            for i in range(A.shape[0]):
                _ = A[i, :]
    else:  # column
        for _ in range(n_trials):
            for j in range(A.shape[1]):
                _ = A[:, j]
    end = time.time()
    return end - start

def plot_reconstruction_error(matrix, max_rank=None):
    """Plot reconstruction error vs rank."""
    U, S, V = torch.linalg.svd(matrix)
    max_rank = len(S) if max_rank is None else max_rank
    errors = []
    
    for k in range(1, max_rank + 1):
        approx = U[:, :k] @ torch.diag(S[:k]) @ V[:k, :]
        error = torch.norm(matrix - approx, p='fro').item()
        errors.append(error)
    
    plt.figure(figsize=(10, 4))
    plt.plot(range(1, max_rank + 1), errors, 'bo-')
    plt.title("Reconstruction Error vs Rank")
    plt.xlabel("Rank")
    plt.ylabel("Frobenius Norm Error")
    plt.grid(True)
    plt.show()

def visualize_svd_components(matrix, k=2):
    """Visualize first k components of SVD."""
    U, S, V = torch.linalg.svd(matrix)
    
    fig, axes = plt.subplots(k, 1, figsize=(12, 4*k))
    if k == 1:
        axes = [axes]
    
    for i in range(k):
        component = torch.outer(U[:, i], V[i, :]) * S[i]
        axes[i].imshow(component.numpy())
        axes[i].set_title(f"Component {i+1} (σ={S[i]:.1f})")
        plt.colorbar(axes[i].imshow(component.numpy()), ax=axes[i])
    
    plt.tight_layout()
    plt.show()

def plot_temperature_pattern(readings):
    """Visualize temperature patterns."""
    plt.figure(figsize=(10, 4))
    times = ['Morning', 'Noon', 'Night']
    for i in range(len(readings)):
        plt.plot(times, readings[i], 'o-', label=f'Day {i+1}')
    plt.grid(True)
    plt.legend()
    plt.title("Daily Temperature Patterns")
    plt.ylabel("Temperature (°C)")
    plt.show()

def visualize_vector_geometry(v1, v2):
    """Visualize vector operations geometrically."""
    plt.figure(figsize=(10, 10))
    
    # Plot vectors
    plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='blue', label='v1')
    plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='red', label='v2')
    
    # Plot projection
    proj = (torch.dot(v1, v2) / torch.dot(v2, v2)) * v2
    plt.quiver(0, 0, proj[0], proj[1], angles='xy', scale_units='xy', scale=1, color='green', label='projection')
    
    # Plot connecting line for projection
    plt.plot([v1[0], proj[0]], [v1[1], proj[1]], '--', color='gray')
    
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.title("Vector Geometry")
    plt.show()

def visualize_cache_hierarchy():
    """Visualize memory hierarchy and cache effects."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Memory hierarchy diagram
    levels = ['L1 Cache\n(64KB)', 'L2 Cache\n(256KB)', 'L3 Cache\n(8MB)', 'Main Memory\n(32GB)']
    sizes = [1, 4, 32, 256]
    latencies = [1, 10, 50, 200]
    
    ax1.barh(levels, sizes, color=['lightblue', 'lightgreen', 'lightyellow', 'lightgray'])
    ax1.set_xscale('log')
    ax1.set_title('Memory Hierarchy (Size)')
    ax1.set_xlabel('Relative Size')
    
    # Cache miss effects
    sizes = np.array([2**i for i in range(10, 25, 2)])
    times = []
    
    for size in sizes:
        A = torch.randn(int(np.sqrt(size)), int(np.sqrt(size)))
        start = time.time()
        _ = A @ A
        times.append(time.time() - start)
    
    ax2.plot(sizes, times, 'bo-')
    ax2.set_xscale('log')
    ax2.set_title('Matrix Multiplication Time vs Size')
    ax2.set_xlabel('Matrix Elements')
    ax2.set_ylabel('Time (s)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_gpu_cpu_comparison(sizes=[1000, 2000, 4000, 8000]):
    """Compare GPU vs CPU performance."""
    cpu_times = []
    gpu_times = []
    
    for size in sizes:
        # CPU timing
        A_cpu = torch.randn(size, size)
        start = time.time()
        _ = A_cpu @ A_cpu
        cpu_times.append(time.time() - start)
        
        # GPU timing (if available)
        if torch.cuda.is_available():
            A_gpu = A_cpu.cuda()
            torch.cuda.synchronize()
            start = time.time()
            _ = A_gpu @ A_gpu
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start)
    
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, cpu_times, 'bo-', label='CPU')
    if gpu_times:
        plt.plot(sizes, gpu_times, 'ro-', label='GPU')
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (s)')
    plt.title('CPU vs GPU Performance')
    plt.legend()
    plt.grid(True)
    plt.show()

def visualize_broadcasting_shapes():
    """Visualize broadcasting rules and shape compatibility."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Case 1: Vector + Scalar
    ax = axes[0,0]
    ax.imshow([[1]], cmap='Blues')
    ax.imshow(np.arange(4).reshape(4,1), extent=[2,3,-0.5,3.5], cmap='Reds', alpha=0.7)
    ax.set_title('Vector + Scalar')
    
    # Case 2: Matrix + Vector
    ax = axes[0,1]
    ax.imshow(np.ones((3,4)), cmap='Blues', alpha=0.5)
    ax.imshow(np.arange(4).reshape(1,4), extent=[-0.5,3.5,2,3], cmap='Reds', alpha=0.7)
    ax.set_title('Matrix + Vector')
    
    # Case 3: 3D + 2D
    ax = axes[1,0]
    ax.text(0.5, 0.5, '3D Tensor\n(2,3,4)', ha='center')
    ax.text(0.5, 0.2, '+', ha='center')
    ax.text(0.5, 0.0, '2D Matrix\n(3,4)', ha='center')
    ax.axis('off')
    
    # Case 4: Invalid broadcasting
    ax = axes[1,1]
    ax.imshow(np.ones((3,4)), cmap='Blues', alpha=0.5)
    ax.imshow(np.arange(3).reshape(3,1), extent=[-0.5,0.5,-0.5,2.5], cmap='Reds', alpha=0.7)
    ax.set_title('Invalid Broadcasting\n(shapes don\'t align)')
    
    plt.tight_layout()
    plt.show()

def plot_memory_bandwidth():
    """Visualize memory bandwidth effects."""
    sizes = [2**i for i in range(10, 25, 2)]
    copy_times = []
    compute_times = []
    
    for size in sizes:
        # Memory copy timing
        x = torch.randn(size)
        start = time.time()
        y = x.clone()
        copy_times.append(time.time() - start)
        
        # Computation timing
        start = time.time()
        y = x * 2
        compute_times.append(time.time() - start)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, copy_times, 'bo-', label='Memory Copy')
    plt.loglog(sizes, compute_times, 'ro-', label='Computation')
    plt.xlabel('Vector Size')
    plt.ylabel('Time (s)')
    plt.title('Memory Bandwidth vs Computation Time')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_convergence(matrix, max_rank=None):
    """Plot error convergence in SVD reconstruction."""
    U, S, V = torch.linalg.svd(matrix)
    max_rank = len(S) if max_rank is None else max_rank
    
    errors = []
    for k in range(1, max_rank + 1):
        approx = U[:,:k] @ torch.diag(S[:k]) @ V[:k,:]
        error = torch.norm(matrix - approx, p='fro').item()
        errors.append(error)
    
    plt.figure(figsize=(10, 4))
    plt.semilogy(range(1, max_rank + 1), errors, 'bo-')
    plt.title("SVD Reconstruction Error")
    plt.xlabel("Rank")
    plt.ylabel("Frobenius Norm Error (log scale)")
    plt.grid(True)
    plt.show() 