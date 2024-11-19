from methods.preliminaries import *

# Is it different for M1 macs? Should I write different code using cuda for the HPC?


# check if we have a GPU available
if pt.backends.mps.is_available():
    mps_device = pt.device("mps")
    x = pt.ones(1,device=mps_device)
    print('MPS backend available:')
    print(x)
else:
    print('No MPS device available')

# check if CUDA is availables
if pt.cuda.is_available():
    cuda_device = pt.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print('CUDA backend available:')
    x = pt.ones(1, device=cuda_device)
    print(x)
else:
    print('No CUDA device available')

device = pt.device('mps' if pt.backends.mps.is_available() else 'cuda' if pt.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

gpu_count = pt.cuda.device_count()
print(f"Number of available GPUs: {gpu_count}")
print()

#Additional info when using cuda
if device.type == 'cuda':
    print(pt.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(pt.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(pt.cuda.memory_reserved(0)/1024**3,1), 'GB')
