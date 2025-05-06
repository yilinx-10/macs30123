# Lab - Week 3 - GPUs and GPU Programming

## Ex1. Sbatch Configurations

```bash
#!/bin/bash

#SBATCH --job-name=gpu_mpi # job name
#SBATCH --output=gpu_mpi.out # output log file
#SBATCH --error=gpu_mpi.err # output error
#SBATCH --nodes=1 #request 1 node
#SBATCH --ntasks-per-node=1 # run 1 task per node, since we have only one task, then same as ntasks = 1
#SBATCH --partition=gpu # we are using gpu
#SBATCH --gres=gpu:1 # one gpu per node
#SBATCH --account=macs30123 # account 

module load python/anaconda-2022.05 cuda/11.7 gcc/10.2.0 # load module

python ./gpu_rand_walk.py 
```

1. Consider the script above. Describe in words what each line of the sbatch script does.
2. Create a file called `lab_wk3.sbatch` on Midway 3 and copy the script above into it. Try to modify the script to accomplish the following (see [the `sbatch` documentation for GPU](https://rcc-uchicago.github.io/user-guide/slurm/sbatch/#gpu-jobs)):
    * Change the number of GPUs requested. What effect might this have on runtime? Would you need to change any of the existing code for it to make a difference? Why?  
            decrease time? need explicitly define how communications occur between gpus.  
            Increasing the number of GPUs requested has the potential to significantly reduce runtime, especially for workloads that can be parallelized and distributed across devices. However, simply requesting more GPUs will not automatically speed up the code unless it is explicitly written to leverage multiple GPUs. This often requires using libraries like torch.nn.DataParallel, torch.distributed, or similar frameworks that enable model/data parallelism. If the current code is written to run only on a single GPU, it would need to be modified to split the workload and manage communication between GPUs for the additional hardware to have any effect. Additionally, the problem size must be large enough to benefit from multiple GPUs. For small or non-GPU-intensive workloads, increasing the number of GPUs might result in minimal or even negative returns due to the overhead of coordinating between devices.

    * Request more CPU cores to drive GPU. Discuss what role this might play in overall performance (if any)?  
            save time?   
            Requesting more CPU cores can help improve GPU performance, but the impact depends on the nature of the workload. CPUs are responsible for tasks such as data preprocessing, loading batches, initiating GPU kernels, and handling memory transfers. If these tasks become a bottleneck—particularly in data-heavy applications like training neural networks—then allocating more CPU cores can help ensure the GPU remains fully utilized. However, if the CPU is not a bottleneck (e.g., the workload is GPU-bound and data loading is already efficient), adding more cores may have little to no effect on performance. In such cases, profiling the workload to identify whether the CPU is a limiting factor is a necessary step before scaling up CPU resources.
    * Add a wall time of 3 minutes to the configurations. Why the wall time may be important for running large jobs?  
            early detection of error? 
3. Submit this job on Midway 3 using the `gpu_rand_walk.py` script from our `in-class-activities` directory this week.

## Ex2. GPU Programming (OpenCL and Map/Reduce)

1. Fill in the code of OpenCL kernels that computes the population variance of a large array...

... either in CUDA:
```python
import pycuda.autoinit
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.curandom as curandom
from pycuda.elementwise import ElementwiseKernel
from pycuda.reduction import ReductionKernel
import numpy as np
import time

# Context and queue are automatically handled by pycuda.autoinit
n_elements = 10 ** 7

# Generate random array on GPU
arr_dev = curandom.rand(n_elements, dtype=np.float32)
# ASK: value of performing computation in single precision vs. double precision?
# chatgpt says better to use single precision. Same for assignment 1? 

# On the GPU: compute the mean of the array, name the kernel `mean_kernel`
mean_kernel = ReductionKernel(dtype_out=np.float32, neutral="0", reduce_expr="a + b", map_expr="x[i] / len(x)", arguments="float *x")

mean_value = mean_kernel(arr_dev).get() / n_elements

# On the GPU: subtract the mean from each element and square the result,
# name the kernel `subtract_and_square_kernel`
subtract_and_square_kernel = ElementwiseKernel("float *arr_dev, float mean_value, float *adjusted_arr_dev",
                                                 "adjusted_arr_dev[i] = (arr_dev[i] - mean_value) ** 2",
                                                 "subtract_and_square_kernel")

adjusted_arr_dev = gpuarray.empty_like(arr_dev)
subtract_and_square_kernel(arr_dev, mean_value, adjusted_arr_dev)

# Write a kernel named `sum_of_squares_kernel` that will compute the sum of `adjusted_arr_dev` on the GPU
sum_of_squares_kernel = ReductionKernel(dtype_out=np.float32, neutral="0", reduce_expr="a + b", 
                                        map_expr="adjusted_arr_dev[i]", arguments="float *adjusted_arr_dev")

# Compute and print the (population) variance on the CPU:
variance_value = sum_of_squares_kernel(adjusted_arr_dev).get() / n_elements

print(f"Mean: {mean_value}, Variance: {variance_value}")
```

Or in OpenCL:
```python
import pyopencl as cl
import pyopencl.clrandom as clrand
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
from pyopencl.reduction import ReductionKernel
import numpy as np
 
ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

n_elements = 10 ** 7
arr_dev = clrand.rand(queue, n_elements, dtype=np.float32)

# On the GPU: compute the mean of the array, name the kernel `mean_kernel`
# YOUR CODE HERE

mean_value = mean_kernel(arr_dev).get() / n_elements

# On the GPU: subtract the mean from each element and square the result,
# name the kernel `subtract_and_square_kernel`
# YOUR CODE HERE

adjusted_arr_dev = cl_array.empty_like(arr_dev)
subtract_and_square_kernel(arr_dev, mean_value, adjusted_arr_dev)

# Write a kernel named `sum_of_squares_kernel` that will compute the sum of `adjusted_arr_dev` on the GPU
# YOUR CODE HERE

# Compute and print the (population) variance on the CPU:
variance_value = sum_of_squares_kernel(adjusted_arr_dev).get() / n_elements

print(f"Mean: {mean_value}, Variance: {variance_value}")
```


2. Write a sbatch script to submit a GPU job that runs the code above on Midway 3.
