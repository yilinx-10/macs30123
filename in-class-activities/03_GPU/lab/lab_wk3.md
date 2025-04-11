# Lab - Week 3 - GPUs and GPU Programming

## Ex1. Sbatch Configurations

```bash
#!/bin/bash

#SBATCH --job-name=gpu_mpi
#SBATCH --output=gpu_mpi.out
#SBATCH --error=gpu_mpi.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=macs30123

module load python/anaconda-2022.05 cuda/11.7 gcc/10.2.0

mpirun python ./gpu_rand_walk.py
```

1. Consider the script above. Describe in words what each line of the sbatch script does.
2. Create a file called `lab_wk3.sbatch` on Midway 3 and copy the script above into it. Try to modify the script to accomplish the following (see [the `sbatch` documentation for GPU](https://rcc-uchicago.github.io/user-guide/slurm/sbatch/#gpu-jobs)):
    * Change the number of GPUs requested. What effect might this have on runtime? Would you need to change any of the existing code for it to make a difference? Why?
    * Request more CPU cores to drive GPU. Discuss what role this might play in overall performance (if any)?
    * Add a wall time of 3 minutes to the configurations. Why the wall time may be important for running large jobs?
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

# On the GPU: compute the mean of the array, name the kernel `mean_kernel`
# YOUR CODE HERE

mean_value = mean_kernel(arr_dev).get() / n_elements

# On the GPU: subtract the mean from each element and square the result,
# name the kernel `subtract_and_square_kernel`
# YOUR CODE HERE

adjusted_arr_dev = gpuarray.empty_like(arr_dev)
subtract_and_square_kernel(arr_dev, mean_value, adjusted_arr_dev)

# Write a kernel named `sum_of_squares_kernel` that will compute the sum of `adjusted_arr_dev` on the GPU
# YOUR CODE HERE

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
