#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>
#include <random>
#include <unistd.h>
#include <curand_kernel.h>
#include "cuda_runtime.h"

#include <cuda/atomic>

// #ifndef VAR_SCOPE
// #define VAR_SCOPE cuda::thread_scope_system
// #endif

// #ifndef GPU_PRODUCER_MEMORY_ORDER
// #define GPU_PRODUCER_MEMORY_ORDER cuda::memory_order_relaxed
// #endif

// #ifndef GPU_CONSUMER_MEMORY_ORDER
// #define GPU_CONSUMER_MEMORY_ORDER cuda::memory_order_relaxed
// #endif

// #ifndef CPU_PRODUCER_MEMORY_ORDER
// #define CPU_PRODUCER_MEMORY_ORDER cuda::memory_order_relaxed
// #endif

// #ifndef CPU_CONSUMER_MEMORY_ORDER
// #define CPU_CONSUMER_MEMORY_ORDER cuda::memory_order_relaxed
// #endif

// #ifndef VAR_SCOPE_X
// #define VAR_SCOPE_X cuda::thread_scope_thread
// #endif

// #ifndef VAR_SCOPE_Y
// #define VAR_SCOPE_Y cuda::thread_scope_device
// #endif

// #ifndef GPU_CONSUMER_LD_X
// #define GPU_CONSUMER_LD_X cuda::memory_order_relaxed
// #endif

// #ifndef GPU_CONSUMER_LD_Y
// #define GPU_CONSUMER_LD_Y cuda::memory_order_relaxed
// #endif

// #ifndef GPU_PRODUCER_ST_X
// #define GPU_PRODUCER_ST_X cuda::memory_order_relaxed
// #endif

// #ifndef GPU_PRODUCER_ST_Y
// #define GPU_PRODUCER_ST_Y cuda::memory_order_relaxed
// #endif

// #ifndef CPU_CONSUMER_LD_X
// #define CPU_CONSUMER_LD_X cuda::memory_order_relaxed
// #endif

// #ifndef CPU_CONSUMER_LD_Y
// #define CPU_CONSUMER_LD_Y cuda::memory_order_relaxed
// #endif

// #ifndef CPU_PRODUCER_ST_X
// #define CPU_PRODUCER_ST_X cuda::memory_order_relaxed
// #endif

// #ifndef CPU_PRODUCER_ST_Y
// #define CPU_PRODUCER_ST_Y cuda::memory_order_relaxed
// #endif

// #define CACHING

#ifdef PRODUCER_FENCE_SCOPE_CTA
#define PRODUCER_FENCE_SCOPE cuda::thread_scope_thread
#elif defined(PRODUCER_FENCE_SCOPE_BLOCK)
#define PRODUCER_FENCE_SCOPE cuda::thread_scope_block
#elif defined(PRODUCER_FENCE_SCOPE_DEV)
#define PRODUCER_FENCE_SCOPE cuda::thread_scope_device
#elif defined(PRODUCER_FENCE_SCOPE_SYS)
#define PRODUCER_FENCE_SCOPE cuda::thread_scope_system
#endif


#ifdef CONSUMER_FENCE_SCOPE_CTA
#define CONSUMER_FENCE_SCOPE cuda::thread_scope_thread
#elif defined(CONSUMER_FENCE_SCOPE_BLOCK)
#define CONSUMER_FENCE_SCOPE cuda::thread_scope_block
#elif defined(CONSUMER_FENCE_SCOPE_DEV)
#define CONSUMER_FENCE_SCOPE cuda::thread_scope_device
#elif defined(CONSUMER_FENCE_SCOPE_SYS)
#define CONSUMER_FENCE_SCOPE cuda::thread_scope_system
#endif


#ifdef RLX_RLX
// #define GPU_PRODUCER_ST_X cuda::memory_order_release
// #define GPU_CONSUMER_LD_X cuda::memory_order_acquire
// #define CPU_PRODUCER_ST_X cuda::memory_order_release
// #define CPU_CONSUMER_LD_X cuda::memory_order_acquire
#define GPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define GPU_PRODUCER_ST_Y cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_Y cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_Y cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_Y cuda::memory_order_relaxed
#elif defined(REL_ACQ)
// #define GPU_PRODUCER_ST_X cuda::memory_order_release
// #define GPU_CONSUMER_LD_X cuda::memory_order_acquire
// #define CPU_PRODUCER_ST_X cuda::memory_order_release
// #define CPU_CONSUMER_LD_X cuda::memory_order_acquire
#define GPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define GPU_PRODUCER_ST_Y cuda::memory_order_release
#define CPU_PRODUCER_ST_Y cuda::memory_order_release
#define GPU_CONSUMER_LD_Y cuda::memory_order_acquire
#define CPU_CONSUMER_LD_Y cuda::memory_order_acquire
#elif defined(RLX_ACQ)
// #define GPU_PRODUCER_ST_X cuda::memory_order_release
// #define GPU_CONSUMER_LD_X cuda::memory_order_acquire
// #define CPU_PRODUCER_ST_X cuda::memory_order_release
// #define CPU_CONSUMER_LD_X cuda::memory_order_acquire
#define GPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define GPU_PRODUCER_ST_Y cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_Y cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_Y cuda::memory_order_acquire
#define CPU_CONSUMER_LD_Y cuda::memory_order_acquire
#elif defined(REL_RLX)
// #define GPU_PRODUCER_ST_X cuda::memory_order_release
// #define GPU_CONSUMER_LD_X cuda::memory_order_acquire
// #define CPU_PRODUCER_ST_X cuda::memory_order_release
// #define CPU_CONSUMER_LD_X cuda::memory_order_acquire
#define GPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define GPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define CPU_PRODUCER_ST_X cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_X cuda::memory_order_relaxed
#define GPU_PRODUCER_ST_Y cuda::memory_order_release
#define CPU_PRODUCER_ST_Y cuda::memory_order_release
#define GPU_CONSUMER_LD_Y cuda::memory_order_relaxed
#define CPU_CONSUMER_LD_Y cuda::memory_order_relaxed
#endif


#ifdef PRODUCER_FENCE_RLX
#define GPU_PRODUCER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_relaxed, PRODUCER_FENCE_SCOPE)
#define CPU_PRODUCER_FENCE() std::atomic_thread_fence(std::memory_order_relaxed)
#elif defined(PRODUCER_FENCE_ACQ_REL)
#define GPU_PRODUCER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, PRODUCER_FENCE_SCOPE)
#define CPU_PRODUCER_FENCE() std::atomic_thread_fence(std::memory_order_acq_rel)
#elif defined(PRODUCER_FENCE_SC)
#define GPU_PRODUCER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_seq_cst, PRODUCER_FENCE_SCOPE)
#define CPU_PRODUCER_FENCE() std::atomic_thread_fence(std::memory_order_seq_cst)
#elif defined(PRODUCER_NO_FENCE)
#define GPU_PRODUCER_FENCE()
#define CPU_PRODUCER_FENCE()
#endif


#ifdef CONSUMER_FENCE_RLX
#define GPU_CONSUMER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_relaxed, CONSUMER_FENCE_SCOPE)
#define CPU_CONSUMER_FENCE() std::atomic_thread_fence(std::memory_order_relaxed)
#elif defined(CONSUMER_FENCE_ACQ_REL)
#define GPU_CONSUMER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_acq_rel, CONSUMER_FENCE_SCOPE)
#define CPU_CONSUMER_FENCE() std::atomic_thread_fence(std::memory_order_acq_rel)
#elif defined(CONSUMER_FENCE_SC)
#define GPU_CONSUMER_FENCE() cuda::atomic_thread_fence(cuda::memory_order_seq_cst, CONSUMER_FENCE_SCOPE)
#define CPU_CONSUMER_FENCE() std::atomic_thread_fence(std::memory_order_seq_cst)
#elif defined(CONSUMER_NO_FENCE) 
#define GPU_CONSUMER_FENCE()
#define CPU_CONSUMER_FENCE()    
#endif

#define PAGE_SIZE 4096

#define DATA_SIZE int

#define TO_STRING(x) #x
#define PRINT_DEFINE(name) std::cout << #name << " = " << TO_STRING(name) << std::endl;

constexpr auto kNumBlocks = 1;
constexpr auto kNumThreads = 64;

#ifdef X_CTA
#define TX bufferElement_t
#elif defined(X_BLOCK)
#define TX bufferElement_b
#elif defined(X_DEV)
#define TX bufferElement_d
#elif defined(X_SYS)
#define TX bufferElement_s
#endif 


#ifdef Y_CTA
#define TY bufferElement_t
#elif defined(Y_BLOCK)
#define TY bufferElement_b
#elif defined(Y_DEV)
#define TY bufferElement_d
#elif defined(Y_SYS)
#define TY bufferElement_s
#endif 


typedef struct bufferElement_t {
    // DATA_SIZE data;
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_t;

typedef struct bufferElement_b {
    // DATA_SIZE data;
    cuda::atomic<DATA_SIZE, cuda::thread_scope_block> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_b;

typedef struct bufferElement_d {
    // DATA_SIZE data;
    cuda::atomic<DATA_SIZE, cuda::thread_scope_thread> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_d;

typedef struct bufferElement_s {
    // DATA_SIZE data;
    cuda::atomic<DATA_SIZE, cuda::thread_scope_system> data;
    char padding[PAGE_SIZE - sizeof(DATA_SIZE)];
} bufferElement_s;


namespace cg = cooperative_groups;

enum class DeviceType {
    CPU,
    GPU
};


void print_usage() {
    std::cout << "Usage: program -i <test_iter> -t <num_threads> -s <memory_stress_array_size>" << std::endl;
}


// static void handle_error(const cudaError_t err, const char* file, const int line) {
//     if (err != cudaSuccess) {
//         printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
//         exit(EXIT_FAILURE);
//     }
// }

// #define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

__device__ void GPUMemoryStress(volatile int* stress_array, volatile int * array_size, volatile cuda::atomic<int, cuda::thread_scope_system> * stop_signal) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int idx = (tid * 17) % *array_size; // Ensure each thread starts differently

    while (stop_signal->load() == 0) {
        int idx1 = idx;
        int idx2 = (idx + stride) % *array_size;

        // Exchange values between two locations
        int temp = stress_array[idx1];
        stress_array[idx1] = stress_array[idx2];
        stress_array[idx2] = temp;

        idx = (idx + stride) % *array_size;
    }
}

void CPUMemoryStress(int tid, volatile int* array, volatile int* stop_thread, int array_size, int num_threads) {
    while (stop_thread[tid] != 1) {
        for (int i = tid + 1; i < array_size - 1; i += num_threads) {
            array[i] = 1 + array[i - 1] + array[i + 1];
        }
    }
}

template <typename S>
__device__ void gpu_consumer_function(TX* volatile com_x, TY* volatile com_y, S* volatile u_com, volatile int* num_iters, cuda::atomic<int, cuda::thread_scope_system>* buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1) {

    #ifdef CACHING
    uint value = 0;

    for (int i = 0; i < *num_iters; i++) {
        value += com_x[i].data.load(cuda::memory_order_relaxed);
        value += com_y[i].data.load(cuda::memory_order_relaxed);
    }
    #endif

    for (int iteration_number = 0; iteration_number < *num_iters; iteration_number++) {
        
        volatile TY* temp_y = &com_y[iteration_number]; // Y
        volatile TX* temp_x = &com_x[iteration_number]; // X

        u_com[iteration_number].fetch_add(1);
        while (u_com[iteration_number] < 2);
    
        int val_y = (int) temp_y->data.load(GPU_CONSUMER_LD_Y); // load Y
    
        GPU_CONSUMER_FENCE();

        int val_x = (int) temp_x->data.load(GPU_CONSUMER_LD_X); // load X

        cuda::atomic_thread_fence(cuda::memory_order_seq_cst, cuda::thread_scope_thread);
    
        buffer_r0[iteration_number].store(val_y, cuda::memory_order_relaxed); // r0 = Y
        buffer_r1[iteration_number].store(val_x, cuda::memory_order_relaxed); // r1 = X
    }
}

template <typename S>
__device__ void gpu_producer_function(TX* volatile com_x, TY* volatile com_y, S* volatile u_com, volatile int* num_iters) {
    
    #ifdef CACHING
    uint value = 0;

    for (volatile int i = 0; i < *num_iters; i++) {
        value += com_x[i].data.load(cuda::memory_order_relaxed);
        value += com_y[i].data.load(cuda::memory_order_relaxed);
    }
    #endif

    for (int iteration_number = 0; iteration_number < *num_iters; iteration_number++) {

        volatile TX* temp_x = &com_x[iteration_number]; // X
        volatile TY* temp_y = &com_y[iteration_number]; // Y

        u_com[iteration_number].fetch_add(1);   

        int val_y = iteration_number + 2; // Y
        int val_x = iteration_number + 1; // X

        while (u_com[iteration_number].load() < 2);

        temp_x->data.store(val_x, GPU_PRODUCER_ST_X); // store X

        GPU_PRODUCER_FENCE();

        temp_y->data.store(val_y, GPU_PRODUCER_ST_Y); // store Y
    }
}

template <typename S>
void cpu_producer_function(volatile TX* com_x, volatile TY* com_y, volatile S* u_com, int num_iters) {

    #ifdef CACHING
    uint value = 0;

    for (int i = 0; i < num_iters; i++) {
        value += com_x[i].data.load(cuda::memory_order_relaxed);
        value += com_y[i].data.load(cuda::memory_order_relaxed);
    }
    #endif

    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {

        volatile TX* temp_x = &com_x[iteration_number]; // X
        volatile TY* temp_y = &com_y[iteration_number]; // Y
        
        u_com[iteration_number].fetch_add(1);

        int val_x = iteration_number + 1;
        int val_y = iteration_number + 2;

        while (u_com[iteration_number].load() < 2);

        temp_x->data.store(val_x, CPU_PRODUCER_ST_X); // store X

        CPU_PRODUCER_FENCE();

        temp_y->data.store(val_y, CPU_PRODUCER_ST_Y); // store Y
    }
}

template <typename S>
void cpu_consumer_function(volatile TX* com_x, volatile TY* com_y, volatile S* u_com, int num_iters, volatile cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, volatile cuda::atomic<int, cuda::thread_scope_system>* buffer_r1) {

    #ifdef CACHING
    uint value = 0;

    for (int i = 0; i < num_iters; i++) {
        value += com_x[i].data.load(cuda::memory_order_relaxed);
        value += com_y[i].data.load(cuda::memory_order_relaxed);
    }
    #endif
    
    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {

        volatile TY* temp_y = &com_y[iteration_number]; // Y
        volatile TX* temp_x = &com_x[iteration_number]; // X

        u_com[iteration_number].fetch_add(1);
        while (u_com[iteration_number].load() < 2);

        int val_y = temp_y->data.load(CPU_CONSUMER_LD_Y); // load Y

        CPU_CONSUMER_FENCE();

        int val_x = temp_x->data.load(CPU_CONSUMER_LD_X); // load X

        std::atomic_thread_fence(std::memory_order_seq_cst);

        buffer_r0[iteration_number].store(val_y, cuda::memory_order_relaxed); // r0 = Y
        buffer_r1[iteration_number].store(val_x, cuda::memory_order_relaxed); // r1 = X
    }

}

template <typename S>
__global__ void main_gpu_kernel_producer(TX * volatile com_x, TY * volatile com_y, S* volatile u_com, volatile int* iteration_number, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, volatile int* memory_stress_array, volatile int * array_size, volatile S * gpu_stop_signal) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        gpu_producer_function(com_x, com_y, u_com, iteration_number);
    } else {
        GPUMemoryStress(memory_stress_array, array_size, gpu_stop_signal);
    }
    
    gpu_stop_signal->fetch_add(1);
}

template <typename S>
__global__ void main_gpu_kernel_consumer(TX* volatile com_x, TY* volatile com_y, S* volatile u_com, volatile int* iteration_number, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, volatile int* memory_stress_array, volatile int * array_size, volatile S * gpu_stop_signal) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        gpu_consumer_function(com_x, com_y, u_com, iteration_number, buffer_r0, buffer_r1);
    } else {
        GPUMemoryStress(memory_stress_array, array_size, gpu_stop_signal);
    }
    
    gpu_stop_signal->fetch_add(1);
}

template <typename S>
__global__ void main_gpu_kernel_producer_consumer(TX* volatile com_x, TY* volatile com_y, S* volatile u_com, volatile int* iteration_number, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, volatile int* memory_stress_array, volatile int * array_size, volatile S * gpu_stop_signal) {
    // int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    
    if (tid == 0 && bid == 0) {
        gpu_producer_function(com_x, com_y, u_com, iteration_number);
    } else if (tid == 0 && bid == 1) {
        gpu_consumer_function(com_x, com_y, u_com, iteration_number, buffer_r0, buffer_r1);
    } else {
        GPUMemoryStress(memory_stress_array, array_size, gpu_stop_signal);
    }
    
    gpu_stop_signal->fetch_add(1);
}

typedef struct {
    int MP_1_0;
    int MP_0_0;
    int MP_0_1;
    int MP_1_1;
} MP_heuristics;


MP_heuristics calculate_heuristics(TX *com_x, TY *com_y, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, int num_iters) {

    MP_heuristics heuristics = {};

    heuristics.MP_1_0 = 0; // typical MP weak behavior (r1 = it + 1, r0 = 0)
    heuristics.MP_0_0 = 0; // MP no propagation behavior (r1 = 0, r0 = 0)
    heuristics.MP_0_1 = 0; // MP interleaved (r1 = 0, r0 = it + 2)
    heuristics.MP_1_1 = 0; // MP successful propagation behavior (r1 = it + 1, r0 = it + 2)

    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {
        if ((buffer_r1[iteration_number].load() == iteration_number + 1) && (buffer_r0[iteration_number].load() == 0)) {
            heuristics.MP_0_1++;
        } else if ((buffer_r1[iteration_number].load() == 0) && (buffer_r0[iteration_number].load() == 0)) {
            heuristics.MP_0_0++;
        } else if ((buffer_r1[iteration_number].load() == 0) && (buffer_r0[iteration_number].load() == iteration_number + 2)) {
            heuristics.MP_1_0++;
        } else if ((buffer_r1[iteration_number].load() == iteration_number + 1) && (buffer_r0[iteration_number].load() == iteration_number + 2)) {
            heuristics.MP_1_1++;
        }
    }

    return heuristics;
}

MP_heuristics add_heuristics(MP_heuristics h1, MP_heuristics h2) {
    MP_heuristics h = {};
    h.MP_1_0 = h1.MP_1_0 + h2.MP_1_0;
    h.MP_0_0 = h1.MP_0_0 + h2.MP_0_0;
    h.MP_0_1 = h1.MP_0_1 + h2.MP_0_1;
    h.MP_1_1 = h1.MP_1_1 + h2.MP_1_1;
    return h;
}

void gpu_producer_cpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, TX * com_x, TY * com_y, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads) {
    int * stop_thread = new int[num_threads];
    // int * gpu_stop_signal = (int *) malloc(sizeof(int));
    cuda::atomic<int, cuda::thread_scope_system> * gpu_stop_signal = (cuda::atomic<int, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<int, cuda::thread_scope_system>));
    gpu_stop_signal->store(0);
    
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::thread stress_threads[num_threads];
    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 0;
    }

    memory_stress_array[0] = 1;
    memory_stress_array[memory_stress_array_size - 1] = 1;

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st] = std::thread(CPUMemoryStress, st, memory_stress_array, stop_thread, memory_stress_array_size, num_threads);
    }

    MP_heuristics total_heuristics = {};

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {

        com_x[iteration_number].data.store(0);
        com_y[iteration_number].data.store(0);

        u_com[iteration_number].store(0);

        buffer_r0[iteration_number].store(0);
        buffer_r1[iteration_number].store(0);

    }
    std::thread main_test_thread;
    main_test_thread = std::thread(cpu_consumer_function<cuda::atomic<int, cuda::thread_scope_system>>, com_x, com_y, u_com, test_iter, buffer_r0, buffer_r1);
    main_gpu_kernel_producer<<<kNumBlocks, kNumThreads>>>(com_x, com_y, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, gpu_stop_signal);

    cudaDeviceSynchronize();
    main_test_thread.join();

    MP_heuristics heuristics = calculate_heuristics(com_x, com_y, buffer_r0, buffer_r1, test_iter);

    #ifdef DEBUG
    std::cout << "GPU-P CPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << std::endl; //<< " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    #endif

    total_heuristics = add_heuristics(total_heuristics, heuristics);

    std::cout << "Total Heuristics GPU-P CPU-C - " << "MP_1_0: " << total_heuristics.MP_1_0 << " | MP_0_0: " << total_heuristics.MP_0_0 << " | MP_0_1: " << total_heuristics.MP_0_1 << " | MP_1_1: " << total_heuristics.MP_1_1 << std::endl;

    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;

}

void cpu_producer_gpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, TX * com_x, TY * com_y, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads) {
    int * stop_thread = new int[num_threads];
    cuda::atomic<int, cuda::thread_scope_system> * gpu_stop_signal = (cuda::atomic<int, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<int, cuda::thread_scope_system>));
    gpu_stop_signal->store(0);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::thread stress_threads[num_threads];
    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 0;
    }

    memory_stress_array[0] = 1;
    memory_stress_array[memory_stress_array_size - 1] = 1;

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st] = std::thread(CPUMemoryStress, st, memory_stress_array, stop_thread, memory_stress_array_size, num_threads);
    }

    MP_heuristics total_heuristics = {};
    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {
        com_x[iteration_number].data.store(0);
        com_y[iteration_number].data.store(0);

        u_com[iteration_number].store(0);
        buffer_r0[iteration_number].store(0);
        buffer_r1[iteration_number].store(0);
    }
    std::thread main_test_thread;
    main_test_thread = std::thread(cpu_producer_function<cuda::atomic<int, cuda::thread_scope_system>>, com_x, com_y, u_com, test_iter);
    main_gpu_kernel_consumer<<<kNumBlocks, kNumThreads>>>(com_x, com_y, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, gpu_stop_signal);

    cudaDeviceSynchronize();
    main_test_thread.join();

    MP_heuristics heuristics = calculate_heuristics(com_x, com_y, buffer_r0, buffer_r1, test_iter);

    #ifdef DEBUG
    std::cout << "CPU-P GPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << std::endl; //" | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    #endif

    total_heuristics = add_heuristics(total_heuristics, heuristics);

    std::cout << "Total Heuristics CPU-P GPU-C - " << "MP_1_0: " << total_heuristics.MP_1_0 << " | MP_0_0: " << total_heuristics.MP_0_0 << " | MP_0_1: " << total_heuristics.MP_0_1 << " | MP_1_1: " << total_heuristics.MP_1_1 << std::endl;


    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;
}

void cpu_producer_cpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, TX * com_x, TY * com_y, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads) {
    int * stop_thread = new int[num_threads];

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::thread stress_threads[num_threads];
    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 0;
    }

    memory_stress_array[0] = 1;
    memory_stress_array[memory_stress_array_size - 1] = 1;

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st] = std::thread(CPUMemoryStress, st, memory_stress_array, stop_thread, memory_stress_array_size, num_threads);
    }

    MP_heuristics total_heuristics = {};

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {

        com_x[iteration_number].data.store(0);
        com_y[iteration_number].data.store(0);

        u_com[iteration_number].store(0);

        buffer_r0[iteration_number].store(0);
        buffer_r1[iteration_number].store(0);
        
    }
    std::thread consumer_thread;
    consumer_thread = std::thread(cpu_consumer_function<cuda::atomic<int, cuda::thread_scope_system>>, com_x, com_y, u_com, test_iter, buffer_r0, buffer_r1);

    std::thread producer_thread;
    producer_thread = std::thread(cpu_producer_function<cuda::atomic<int, cuda::thread_scope_system>>, com_x, com_y, u_com, test_iter);

    producer_thread.join();
    consumer_thread.join();

    MP_heuristics heuristics = calculate_heuristics(com_x, com_y, buffer_r0, buffer_r1, test_iter);

    #ifdef DEBUG
    std::cout << "CPU-P CPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << std::endl; // " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    #endif

    total_heuristics = add_heuristics(total_heuristics, heuristics);

    std::cout << "Total Heuristics CPU-P CPU-C - " << "MP_1_0: " << total_heuristics.MP_1_0 << " | MP_0_0: " << total_heuristics.MP_0_0 << " | MP_0_1: " << total_heuristics.MP_0_1 << " | MP_1_1: " << total_heuristics.MP_1_1 << std::endl;

    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;

}

void gpu_producer_gpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, TX * com_x, TY * com_y, cuda::atomic<int, cuda::thread_scope_system> * buffer_r0, cuda::atomic<int, cuda::thread_scope_system> * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads) {
    int * stop_thread = new int[num_threads];
    cuda::atomic<int, cuda::thread_scope_system> * gpu_stop_signal = (cuda::atomic<int, cuda::thread_scope_system> *) malloc(sizeof(cuda::atomic<int, cuda::thread_scope_system>));
    gpu_stop_signal->store(0);

    std::thread stress_threads[num_threads];
    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 0;
    }

    memory_stress_array[0] = 1;
    memory_stress_array[memory_stress_array_size - 1] = 1;

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st] = std::thread(CPUMemoryStress, st, memory_stress_array, stop_thread, memory_stress_array_size, num_threads);
    }

    MP_heuristics total_heuristics = {};

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {

        com_x[iteration_number].data.store(0);
        com_y[iteration_number].data.store(0);

        u_com[iteration_number].store(0);

        buffer_r0[iteration_number].store(0);
        buffer_r1[iteration_number].store(0);
    }

    main_gpu_kernel_producer_consumer<<<kNumBlocks + 1, kNumThreads, 0>>>(com_x, com_y, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, gpu_stop_signal);
    
    cudaDeviceSynchronize();

    MP_heuristics heuristics = calculate_heuristics(com_x, com_y, buffer_r0, buffer_r1, test_iter);

    #ifdef DEBUG
    std::cout << "GPU-P GPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << std::endl; //" | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    #endif

    total_heuristics = add_heuristics(total_heuristics, heuristics);

    std::cout << "Total Heuristics GPU-P GPU-C - " << "MP_1_0: " << total_heuristics.MP_1_0 << " | MP_0_0: " << total_heuristics.MP_0_0 << " | MP_0_1: " << total_heuristics.MP_0_1 << " | MP_1_1: " << total_heuristics.MP_1_1 << std::endl;


    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;
}

int main(int argc, char** argv) {
    int test_iter = 0;
    int num_threads = 0;
    int memory_stress_array_size = 0;

    int opt;
    while ((opt = getopt(argc, argv, "i:t:s:c:o:")) != -1) {
        switch (opt) {
            case 'i':
                test_iter = std::stoi(optarg);
                break;
            case 't':
                num_threads = std::stoi(optarg);
                break;
            case 's':
                memory_stress_array_size = std::stoi(optarg);
                break;
            default:
                print_usage();
                return -1;
        }
    }

    if (test_iter == 0 || num_threads == 0 || memory_stress_array_size == 0) {
        print_usage();
        return -1;
    }

    std::cout << "Number of Iterations: " << test_iter << std::endl;
    std::cout << "Page Size: " << sizeof(bufferElement_s) << " bytes" << std::endl;


    int totalSize = test_iter;

    std::cout << "Array Size: " << totalSize << " elements, " << totalSize * sizeof(TX) << " bytes" << std::endl;

    std::cout << "Memory Stress Array Size: " << memory_stress_array_size << std::endl;
    std::cout << "Stress Threads: " << num_threads << std::endl;

    // PRINT_DEFINE(CPU_CONSUMER_LD_X);
    // PRINT_DEFINE(CPU_CONSUMER_LD_Y);
    // PRINT_DEFINE(GPU_CONSUMER_LD_X);
    // PRINT_DEFINE(GPU_CONSUMER_LD_Y);
    // PRINT_DEFINE(CPU_PRODUCER_ST_X);
    // PRINT_DEFINE(CPU_PRODUCER_ST_Y);
    // PRINT_DEFINE(GPU_PRODUCER_ST_X);
    // PRINT_DEFINE(GPU_PRODUCER_ST_Y);

    // PRINT_DEFINE(TX);
    // PRINT_DEFINE(TY);

    cuda::atomic<int, cuda::thread_scope_system>* u_com = nullptr;
    TX* com_x = nullptr;
    TY* com_y = nullptr;

    cuda::atomic<int, cuda::thread_scope_system> *buffer_r0 = nullptr; // r0 stores Y
    cuda::atomic<int, cuda::thread_scope_system> *buffer_r1 = nullptr; // r1 stores X


    u_com = (cuda::atomic<int, cuda::thread_scope_system>*) malloc(test_iter * sizeof(cuda::atomic<int, cuda::thread_scope_system>));
  
    com_x = (TX*) malloc(totalSize * sizeof(TX));
    com_y = (TY*) malloc(totalSize * sizeof(TY));

    buffer_r0 = (cuda::atomic<int, cuda::thread_scope_system>*) malloc(test_iter * sizeof(cuda::atomic<int, cuda::thread_scope_system>));
    buffer_r1 = (cuda::atomic<int, cuda::thread_scope_system>*) malloc(test_iter * sizeof(cuda::atomic<int, cuda::thread_scope_system>));


    int * memory_stress_array = (int*) malloc(memory_stress_array_size * sizeof(int));


    std::cout << std::endl;

    cpu_producer_cpu_consumer(u_com, com_x, com_y, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads);
    gpu_producer_gpu_consumer(u_com, com_x, com_y, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads);
    gpu_producer_cpu_consumer(u_com, com_x, com_y, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads);
    cpu_producer_gpu_consumer(u_com, com_x, com_y, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads);

    free(u_com);
    free(com_x);
    free(com_y);
    free(buffer_r0);
    free(buffer_r1);
    free(memory_stress_array);

    return 0;
}