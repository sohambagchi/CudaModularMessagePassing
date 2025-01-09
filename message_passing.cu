#include <cooperative_groups.h>
#include <device_launch_parameters.h>

#include <iostream>
#include <atomic>
#include <chrono>
#include <thread>
#include <random>
#include <unistd.h>
#include "cuda_runtime.h"

#include <cuda/atomic>

#ifndef VAR_SCOPE
#define VAR_SCOPE cuda::thread_scope_system
#endif

#ifndef GPU_PRODUCER_MEMORY_ORDER
#define GPU_PRODUCER_MEMORY_ORDER cuda::memory_order_relaxed
#endif

#ifndef GPU_CONSUMER_MEMORY_ORDER
#define GPU_CONSUMER_MEMORY_ORDER cuda::memory_order_relaxed
#endif

#ifndef CPU_PRODUCER_MEMORY_ORDER
#define CPU_PRODUCER_MEMORY_ORDER std::memory_order_relaxed
#endif

#ifndef CPU_CONSUMER_MEMORY_ORDER
#define CPU_CONSUMER_MEMORY_ORDER std::memory_order_relaxed
#endif


#define TO_STRING(x) #x
#define PRINT_DEFINE(name) std::cout << #name << " = " << TO_STRING(name) << std::endl;

constexpr auto kNumBlocks = 1;
constexpr auto kNumThreads = 128;


namespace cg = cooperative_groups;

enum class DeviceType {
    CPU,
    GPU
};


void print_usage() {
    std::cout << "Usage: program -i <test_iter> -t <num_threads> -s <memory_stress_array_size> -c <cacheline> -o <same_or_different> -p <producer> -u <consumer>" << std::endl;
}


static void handle_error(const cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handle_error(err, __FILE__, __LINE__))

__device__ void GPUMemoryStress(volatile int* stress_array, int tid, volatile int * array_size) {
    for (int i = 0; i < *array_size; i++) {
        int x = stress_array[tid + 1];
        stress_array[tid] = 90;
    }
    return; //TODO:
}

void CPUMemoryStress(int tid, volatile int* array, volatile int* stop_thread, int array_size, int num_threads) {
    while (stop_thread[tid] != 1) {
        for (int i = tid + 1; i < array_size - 1; i += num_threads) {
            array[i] = 1 + array[i - 1] + array[i + 1];
        }
    }
}

template <typename T, typename S>
__device__ void gpu_consumer_function(T* volatile com, S* volatile u_com, volatile int* num_iters, int* buffer_r0, int * buffer_r1, volatile int * X_offset, volatile int * Y_offset) {

    for (int iteration_number = 0; iteration_number < *num_iters; iteration_number++) {
        
        volatile T* temp_2 = com + (iteration_number * *X_offset + *Y_offset); // Y
        volatile T* temp_3 = com + (iteration_number * *X_offset); // X

        atomicAdd((int*)&u_com[iteration_number], 1);
        while (u_com[iteration_number] < 2);
    
        int val_2 = (int) temp_2->load(GPU_CONSUMER_MEMORY_ORDER); // load Y
    
        #ifdef GPU_CONSUMER_FENCE_ORDER 
        cuda::atomic_thread_fence(GPU_CONSUMER_FENCE_ORDER, GPU_CONSUMER_FENCE_SCOPE);
        #endif

        int val_3 = (int) temp_3->load(GPU_CONSUMER_MEMORY_ORDER); // load X
    
        buffer_r0[iteration_number] = val_2; // r0 = Y
        buffer_r1[iteration_number] = val_3; // r1 = X
    }
}

template <typename T, typename S>
__device__ void gpu_producer_function(T* volatile com, S* volatile u_com, volatile int* num_iters, volatile int* X_offset, volatile int* Y_offset) {
    
    for (int iteration_number = 0; iteration_number < *num_iters; iteration_number++) {
        // u_com[0] = iteration_number;
        volatile T* temp_2 = com + (iteration_number * *X_offset); // X
        volatile T* temp_3 = com + (iteration_number * *X_offset + *Y_offset); // Y

        atomicAdd((int*)&u_com[iteration_number], 1);
        int val_3 = iteration_number + 2; // Y
        int val_2 = iteration_number + 1; // X
        while (u_com[iteration_number] < 2);
        temp_2->store(val_2, GPU_PRODUCER_MEMORY_ORDER); // store X

        #ifdef GPU_PRODUCER_FENCE_ORDER
        cuda::atomic_thread_fence(GPU_PRODUCER_FENCE_ORDER, GPU_PRODUCER_FENCE_SCOPE);
        #endif

        temp_3->store(val_3, GPU_PRODUCER_MEMORY_ORDER); // store Y
    }
}

template <typename T, typename S>
void cpu_producer_function(volatile T* com, volatile S* u_com, int num_iters, volatile int X_offset, volatile int Y_offset) {

    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {
        // u_com[0] = iteration_number;
        volatile T* temp_0 = com + (iteration_number * X_offset); // X
        volatile T* temp_1 = com + (iteration_number * X_offset + Y_offset); // Y
        
        u_com[iteration_number].fetch_add(1);
        int val_0 = iteration_number + 1;
        int val_1 = iteration_number + 2;
        while (u_com[iteration_number] < 2);

        temp_0->store(val_0, CPU_PRODUCER_MEMORY_ORDER); // store X

        #ifdef CPU_PRODUCER_FENCE_ORDER
        std::atomic_thread_fence(CPU_PRODUCER_FENCE_ORDER);
        #endif

        temp_1->store(val_1, CPU_PRODUCER_MEMORY_ORDER); // store Y
    }
}

template <typename T, typename S>
void cpu_consumer_function(volatile T* com, volatile S* u_com, int num_iters, volatile int* buffer_r0, volatile int* buffer_r1, volatile int X_offset, volatile int Y_offset) {

    
    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {
        volatile T* temp_2 = com + (iteration_number * X_offset + Y_offset); // Y
        volatile T* temp_3 = com + (iteration_number * X_offset); // X
        u_com[iteration_number].fetch_add(1);
        while (u_com[iteration_number] < 2);

        int val_2 = temp_2->load(CPU_CONSUMER_MEMORY_ORDER); // load Y

        #ifdef CPU_CONSUMER_FENCE_ORDER
        std::atomic_thread_fence(CPU_CONSUMER_FENCE_ORDER);
        #endif

        int val_3 = temp_3->load(CPU_CONSUMER_MEMORY_ORDER); // load X

        buffer_r0[iteration_number] = val_2; // r0 = Y
        buffer_r1[iteration_number] = val_3; // r1 = X
    }

}

template <typename T, typename S>
__global__ void main_gpu_kernel_producer(T* volatile com, S* volatile u_com, volatile int* iteration_number, int * buffer_r0, int * buffer_r1, volatile int* memory_stress_array, volatile int * array_size, volatile int * X_offset, volatile int * Y_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        gpu_producer_function(com, u_com, iteration_number, X_offset, Y_offset);
    } else {
        GPUMemoryStress(memory_stress_array, tid, array_size);
    }
}

template <typename T, typename S>
__global__ void main_gpu_kernel_consumer(T* volatile com, S* volatile u_com, volatile int* iteration_number, int * buffer_r0, int * buffer_r1, volatile int* memory_stress_array, volatile int * array_size, volatile int * X_offset, volatile int * Y_offset) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid == 0) {
        gpu_consumer_function(com, u_com, iteration_number, buffer_r0, buffer_r1, X_offset, Y_offset);
    } else {
        GPUMemoryStress(memory_stress_array, tid, array_size);
    }
}

typedef struct {
    int MP_1_0;
    int MP_0_0;
    int MP_0_1;
    int MP_1_1;
} MP_heuristics;

template <typename T> 
MP_heuristics calculate_heuristics(T *com, int message_pos_X, int message_pos_Y, int * buffer_r0, int * buffer_r1, int num_iters) {

    MP_heuristics heuristics = {};

    heuristics.MP_1_0 = 0; // typical MP weak behavior (r1 = it + 1, r0 = 0)
    heuristics.MP_0_0 = 0; // MP no propagation behavior (r1 = 0, r0 = 0)
    heuristics.MP_0_1 = 0; // MP interleaved (r1 = 0, r0 = it + 2)
    heuristics.MP_1_1 = 0; // MP successful propagation behavior (r1 = it + 1, r0 = it + 2)
    for (int iteration_number = 0; iteration_number < num_iters; iteration_number++) {
        if ((buffer_r1[iteration_number] == iteration_number + 1) && (buffer_r0[iteration_number] == 0)) {
            heuristics.MP_0_0++;
        }

        if ((buffer_r1[iteration_number] == 0) && (buffer_r0[iteration_number] == 0)) {
            heuristics.MP_0_0++;
        }

        if ((buffer_r1[iteration_number] == 0) && (buffer_r0[iteration_number] == iteration_number + 2)) {
            heuristics.MP_0_1++;
        }

        if ((buffer_r1[iteration_number] == iteration_number + 1) && (buffer_r0[iteration_number] == iteration_number + 2)) {
            heuristics.MP_1_1++;
        }
    }

    return heuristics;
}

void gpu_producer_cpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, cuda::atomic<int, VAR_SCOPE> * com, int * buffer_r0, int * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads, int * X_offset, int * Y_offset, int totalSize) {
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

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {
        com[iteration_number * *X_offset] = 0;
        com[iteration_number * *X_offset + *Y_offset] = 0;
        u_com[iteration_number].store(0);
    }

    std::thread main_test_thread;
    main_test_thread = std::thread(cpu_consumer_function<cuda::atomic<int, VAR_SCOPE>, cuda::atomic<int, cuda::thread_scope_system>>, com, u_com, test_iter, buffer_r0, buffer_r1, *X_offset, *Y_offset);
    main_gpu_kernel_producer<<<kNumBlocks, kNumThreads>>>(com, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, X_offset, Y_offset);

    cudaDeviceSynchronize();
    main_test_thread.join();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // std::cout << std::endl;

    MP_heuristics heuristics = calculate_heuristics(com, *X_offset, *Y_offset, buffer_r0, buffer_r1, test_iter);

    std::cout << "GPU-P CPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;

}

void cpu_producer_gpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, cuda::atomic<int, VAR_SCOPE> * com, int * buffer_r0, int * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads, int * X_offset, int * Y_offset, int totalSize) {
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

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {
        com[iteration_number * *X_offset] = 0;
        com[iteration_number * *X_offset + *Y_offset] = 0;
        u_com[iteration_number].store(0);
    }

    std::thread main_test_thread;
    main_test_thread = std::thread(cpu_producer_function<cuda::atomic<int, VAR_SCOPE>, cuda::atomic<int, cuda::thread_scope_system>>, com, u_com, test_iter, *X_offset, *Y_offset);
    main_gpu_kernel_consumer<<<kNumBlocks, kNumThreads>>>(com, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, X_offset, Y_offset);

    cudaDeviceSynchronize();
    main_test_thread.join();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // std::cout << std::endl;

    MP_heuristics heuristics = calculate_heuristics(com, *X_offset, *Y_offset, buffer_r0, buffer_r1, test_iter);

    std::cout << "CPU-P GPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;
}

void cpu_producer_cpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, cuda::atomic<int, VAR_SCOPE> * com, int * buffer_r0, int * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads, int * X_offset, int * Y_offset, int totalSize) {
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

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {
        com[iteration_number * *X_offset] = 0;
        com[iteration_number * *X_offset + *Y_offset] = 0;
        u_com[iteration_number].store(0);
    }

    std::thread consumer_thread;
    consumer_thread = std::thread(cpu_consumer_function<cuda::atomic<int, VAR_SCOPE>, cuda::atomic<int, cuda::thread_scope_system>>, com, u_com, test_iter, buffer_r0, buffer_r1, *X_offset, *Y_offset);

    std::thread producer_thread;
    producer_thread = std::thread(cpu_producer_function<cuda::atomic<int, VAR_SCOPE>, cuda::atomic<int, cuda::thread_scope_system>>, com, u_com, test_iter, *X_offset, *Y_offset);

    producer_thread.join();
    consumer_thread.join();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // std::cout << std::endl;

    MP_heuristics heuristics = calculate_heuristics(com, *X_offset, *Y_offset, buffer_r0, buffer_r1, test_iter);

    std::cout << "CPU-P CPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    for (int st = 0; st < num_threads; st++) {
        stop_thread[st] = 1;
    }

    for (int st = 0; st < num_threads; st++) {
        stress_threads[st].join();
    }

    return;

}

void gpu_producer_gpu_consumer(cuda::atomic<int, cuda::thread_scope_system> * u_com, cuda::atomic<int, VAR_SCOPE> * com, int * buffer_r0, int * buffer_r1, int * memory_stress_array, int memory_stress_array_size, int test_iter, int num_threads, int * X_offset, int * Y_offset, int totalSize) {
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

    for (int iteration_number = 0; iteration_number < test_iter; iteration_number++) {
        com[iteration_number * *X_offset] = 0;
        com[iteration_number * *X_offset + *Y_offset] = 0;
        u_com[iteration_number].store(0);
    }

    cudaStream_t stream_a, stream_b;
    cudaStreamCreate(&stream_a);
    cudaStreamCreate(&stream_b);

    main_gpu_kernel_consumer<<<kNumBlocks, kNumThreads, 0, stream_a>>>(com, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, X_offset, Y_offset);
    main_gpu_kernel_producer<<<kNumBlocks, kNumThreads, 0, stream_b>>>(com, u_com, &test_iter, buffer_r0, buffer_r1, memory_stress_array, &memory_stress_array_size, X_offset, Y_offset);

    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    // std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << std::endl;
    // std::cout << std::endl;

    MP_heuristics heuristics = calculate_heuristics(com, *X_offset, *Y_offset, buffer_r0, buffer_r1, test_iter);

    std::cout << "GPU-P GPU-C | MP_1_0: " << heuristics.MP_1_0 << " | MP_0_0: " << heuristics.MP_0_0 << " | MP_0_1: " << heuristics.MP_0_1 << " | MP_1_1: " << heuristics.MP_1_1 << " | Time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

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
    int cacheline_size;
    int offset;
    // DeviceType producer;
    // DeviceType consumer;

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
            case 'c':
                if (strcmp(optarg, "gpu") == 0) {
                    cacheline_size = 128;
                } else if (strcmp(optarg, "cpu") == 0) {
                    cacheline_size = 64;
                } else {
                    std::cerr << "Invalid cacheline specified." << optarg << std::endl;
                    return -1;
                }
                break;
            case 'o':
                if (strcmp(optarg, "same") == 0) {
                    offset = 0;
                } else {
                    offset = cacheline_size;
                }
                break;
            // case 'p':
            //     if (strcmp(optarg, "gpu") == 0) {
            //         producer = DeviceType::GPU;
            //     } else if (strcmp(optarg, "cpu") == 0) {
            //         producer = DeviceType::CPU;
            //     } else {
            //         std::cerr << "Invalid producer specified." << std::endl;
            //         return -1;
            //     }
            //     break;
            // case 'u':
            //     if (strcmp(optarg, "gpu") == 0) {
            //         consumer = DeviceType::GPU;
            //     } else if (strcmp(optarg, "cpu") == 0) {
            //         consumer = DeviceType::CPU;
            //     } else {
            //         std::cerr << "Invalid consumer specified." << std::endl;
            //         return -1;
            //     }
            //     break;
            default:
                print_usage();
                return -1;
        }
    }

    if (test_iter == 0 || num_threads == 0 || memory_stress_array_size == 0 || cacheline_size == 0) {
        print_usage();
        return -1;
    }

    std::cout << "Number of Iterations: " << test_iter << std::endl;
    std::cout << "Cacheline Size: " << cacheline_size << (offset == 0 ? "B | Same Cacheline" : " | Different Cacheline") << std::endl;

    int totalSize = 0;
    if (offset == 0) {
        totalSize = test_iter * cacheline_size / sizeof(cuda::atomic<int, VAR_SCOPE>);
    } else {
        totalSize = test_iter * 2 * cacheline_size / sizeof(cuda::atomic<int, VAR_SCOPE>);
    }

    std::cout << "Array Size: " << totalSize << " elements, " << totalSize * sizeof(cuda::atomic<int, VAR_SCOPE>) << " bytes" << std::endl;

    std::cout << "Memory Stress Array Size: " << memory_stress_array_size << std::endl;
    std::cout << "Stress Threads: " << num_threads << std::endl;

    PRINT_DEFINE(VAR_SCOPE);
    PRINT_DEFINE(GPU_PRODUCER_MEMORY_ORDER);
    PRINT_DEFINE(GPU_CONSUMER_MEMORY_ORDER);
    PRINT_DEFINE(CPU_PRODUCER_MEMORY_ORDER);
    PRINT_DEFINE(CPU_CONSUMER_MEMORY_ORDER);
    
    #ifdef GPU_PRODUCER_FENCE_SCOPE
    PRINT_DEFINE(GPU_PRODUCER_FENCE_SCOPE);
    #endif

    #ifdef GPU_CONSUMER_FENCE_SCOPE
    PRINT_DEFINE(GPU_CONSUMER_FENCE_SCOPE);
    #endif

    #ifdef CPU_PRODUCER_FENCE_ORDER
    PRINT_DEFINE(CPU_PRODUCER_FENCE_ORDER);
    #endif

    #ifdef CPU_CONSUMER_FENCE_ORDER
    PRINT_DEFINE(CPU_CONSUMER_FENCE_ORDER);
    #endif

    #ifdef GPU_PRODUCER_FENCE_ORDER
    PRINT_DEFINE(GPU_PRODUCER_FENCE_ORDER);
    #endif

    #ifdef GPU_CONSUMER_FENCE_ORDER
    PRINT_DEFINE(GPU_CONSUMER_FENCE_ORDER);
    #endif

    cuda::atomic<int, cuda::thread_scope_system>* u_com = nullptr;
    // cuda::atomic<int, cuda::thread_scope_device>* u_com = nullptr;
    cuda::atomic<int, VAR_SCOPE>* com = nullptr;

    int *buffer_r0 = nullptr; // r0 stores Y
    int *buffer_r1 = nullptr; // r1 stores X

    // int *iteration = nullptr;

    int *X_offset = nullptr;
    int *Y_offset = nullptr;

    u_com = (cuda::atomic<int, cuda::thread_scope_system>*) malloc(test_iter * sizeof(cuda::atomic<int, cuda::thread_scope_system>));
    // u_com = (cuda::atomic<int, VAR_SCOPE>*) malloc(test_iter * sizeof(cuda::atomic<int, VAR_SCOPE>));
    com = (cuda::atomic<int, VAR_SCOPE>*) malloc(totalSize * sizeof(cuda::atomic<int, VAR_SCOPE>));

    buffer_r0 = (int*) malloc(test_iter * sizeof(int));
    buffer_r1 = (int*) malloc(test_iter * sizeof(int));

    // iteration = (int*) malloc(sizeof(int));

    int * memory_stress_array = (int*) malloc(memory_stress_array_size * sizeof(int));

    X_offset = (int*) malloc(sizeof(int));
    Y_offset = (int*) malloc(sizeof(int));

    *X_offset = cacheline_size / sizeof(cuda::atomic<int, VAR_SCOPE>);
    *Y_offset = offset == 0 ? *X_offset / 2 : *X_offset;

    std::cout << std::endl;

    gpu_producer_cpu_consumer(u_com, com, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads, X_offset, Y_offset, totalSize);
    cpu_producer_gpu_consumer(u_com, com, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads, X_offset, Y_offset, totalSize);
    gpu_producer_gpu_consumer(u_com, com, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads, X_offset, Y_offset, totalSize);
    cpu_producer_cpu_consumer(u_com, com, buffer_r0, buffer_r1, memory_stress_array, memory_stress_array_size, test_iter, num_threads, X_offset, Y_offset, totalSize);

    free(u_com);
    free(com);
    free(buffer_r0);
    free(buffer_r1);
    free(memory_stress_array);
    free(X_offset);
    free(Y_offset);

    return 0;
}