import os
import itertools
import threading

NVCC = os.popen('which nvcc').read().strip()
NVCCFLAGS = '-std=c++14 --expt-relaxed-constexpr -arch=sm_87'

OUTPUT = 'MP'
PTX = 'MP_PTX'

SRCS = 'message_passing.cu'

VAR_SCOPES = [
    'cuda::thread_scope_thread', 'cuda::thread_scope_block',
    'cuda::thread_scope_device', 'cuda::thread_scope_system'
]

GPU_PRODUCER_MEMORY_ORDERS = ['cuda::memory_order_relaxed', 'cuda::memory_order_release']
GPU_CONSUMER_MEMORY_ORDERS = ['cuda::memory_order_relaxed', 'cuda::memory_order_acquire']
CPU_PRODUCER_MEMORY_ORDERS = ['cuda::memory_order_relaxed', 'cuda::memory_order_release']
CPU_CONSUMER_MEMORY_ORDERS = ['cuda::memory_order_relaxed', 'cuda::memory_order_acquire']
# CPU_PRODUCER_MEMORY_ORDERS = ['std::memory_order_relaxed', 'std::memory_order_release']
# CPU_CONSUMER_MEMORY_ORDERS = ['std::memory_order_relaxed', 'std::memory_order_acquire']

GPU_CONSUMER_FENCE_SCOPES = [
    'cuda::thread_scope_system', 'cuda::thread_scope_device',
    'cuda::thread_scope_block', 'cuda::thread_scope_thread'
]
GPU_PRODUCER_FENCE_SCOPES = [
    'cuda::thread_scope_system', 'cuda::thread_scope_device',
    'cuda::thread_scope_block', 'cuda::thread_scope_thread'
]

GPU_PRODUCER_FENCE_ORDERS = [
    'cuda::memory_order_relaxed', 'cuda::memory_order_release',
    'cuda::memory_order_acq_rel', 'cuda::memory_order_seq_cst'
]
GPU_CONSUMER_FENCE_ORDERS = [
    'cuda::memory_order_relaxed', 'cuda::memory_order_acquire',
    'cuda::memory_order_acq_rel', 'cuda::memory_order_seq_cst'
]

CPU_PRODUCER_FENCE_ORDERS = [
    'std::memory_order_relaxed', 'std::memory_order_release',
    'std::memory_order_acq_rel', 'std::memory_order_seq_cst'
]
CPU_CONSUMER_FENCE_ORDERS = [
    'std::memory_order_relaxed', 'std::memory_order_acquire',
    'std::memory_order_acq_rel', 'std::memory_order_seq_cst'
]

def subst_filename(name):
    return name.replace('cuda::', '').replace('std::', '').replace('memory_order_', '').replace('thread_scope_', '').replace('seq_cst', 'seqcst').replace('acq_rel', 'acqrel')

def make_target(var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order, gpu_consumer_fence_scope, gpu_producer_fence_scope, gpu_producer_fence_order, gpu_consumer_fence_order, cpu_producer_fence_order, cpu_consumer_fence_order):
    filename = f"{OUTPUT}_{subst_filename(f'{var_scope}_{gpu_producer_memory_order}_{gpu_consumer_memory_order}_{cpu_producer_memory_order}_{cpu_consumer_memory_order}_{gpu_consumer_fence_scope}_{gpu_producer_fence_scope}_{gpu_producer_fence_order}_{gpu_consumer_fence_order}_{cpu_producer_fence_order}_{cpu_consumer_fence_order}')}.out"
    command = f"{NVCC} {NVCCFLAGS} -DVAR_SCOPE={var_scope} -DGPU_PRODUCER_MEMORY_ORDER={gpu_producer_memory_order} -DGPU_CONSUMER_MEMORY_ORDER={gpu_consumer_memory_order} -DCPU_PRODUCER_MEMORY_ORDER={cpu_producer_memory_order} -DCPU_CONSUMER_MEMORY_ORDER={cpu_consumer_memory_order} -DGPU_CONSUMER_FENCE_SCOPE={gpu_consumer_fence_scope} -DGPU_PRODUCER_FENCE_SCOPE={gpu_producer_fence_scope} -DGPU_PRODUCER_FENCE_ORDER={gpu_producer_fence_order} -DGPU_CONSUMER_FENCE_ORDER={gpu_consumer_fence_order} -DCPU_PRODUCER_FENCE_ORDER={cpu_producer_fence_order} -DCPU_CONSUMER_FENCE_ORDER={cpu_consumer_fence_order} -o {filename} {SRCS}"
    os.system(command)

def make_target_no_fence(var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order):
    filename = f"{OUTPUT}_{subst_filename(f'{var_scope}_{gpu_producer_memory_order}_{gpu_consumer_memory_order}_{cpu_producer_memory_order}_{cpu_consumer_memory_order}')}.out"
    command = f"{NVCC} {NVCCFLAGS} -DVAR_SCOPE={var_scope} -DGPU_PRODUCER_MEMORY_ORDER={gpu_producer_memory_order} -DGPU_CONSUMER_MEMORY_ORDER={gpu_consumer_memory_order} -DCPU_PRODUCER_MEMORY_ORDER={cpu_producer_memory_order} -DCPU_CONSUMER_MEMORY_ORDER={cpu_consumer_memory_order} -o {filename} {SRCS}"
    os.system(command)

def launch_in_thread(target, *args):
    thread = threading.Thread(target=target, args=args)
    thread.start()
    return thread

threads = []

for var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order, gpu_consumer_fence_scope, gpu_producer_fence_scope, gpu_producer_fence_order, gpu_consumer_fence_order, cpu_producer_fence_order, cpu_consumer_fence_order in itertools.product(
    VAR_SCOPES, GPU_PRODUCER_MEMORY_ORDERS, GPU_CONSUMER_MEMORY_ORDERS, CPU_PRODUCER_MEMORY_ORDERS, CPU_CONSUMER_MEMORY_ORDERS, GPU_CONSUMER_FENCE_SCOPES, GPU_PRODUCER_FENCE_SCOPES, GPU_PRODUCER_FENCE_ORDERS, GPU_CONSUMER_FENCE_ORDERS, CPU_PRODUCER_FENCE_ORDERS, CPU_CONSUMER_FENCE_ORDERS):
    threads.append(launch_in_thread(make_target, var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order, gpu_consumer_fence_scope, gpu_producer_fence_scope, gpu_producer_fence_order, gpu_consumer_fence_order, cpu_producer_fence_order, cpu_consumer_fence_order))
    # make_target(var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order, gpu_consumer_fence_scope, gpu_producer_fence_scope, gpu_producer_fence_order, gpu_consumer_fence_order, cpu_producer_fence_order, cpu_consumer_fence_order)

for var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order in itertools.product(
    VAR_SCOPES, GPU_PRODUCER_MEMORY_ORDERS, GPU_CONSUMER_MEMORY_ORDERS, CPU_PRODUCER_MEMORY_ORDERS, CPU_CONSUMER_MEMORY_ORDERS):
    threads.append(launch_in_thread(make_target_no_fence, var_scope, gpu_producer_memory_order, gpu_consumer_memory_order, cpu_producer_memory_order, cpu_consumer_memory_order))

for thread in threads:
    thread.join()

def clean():
    os.system('rm -f *.out *.ptx *.sass')

if __name__ == '__main__':
    clean()