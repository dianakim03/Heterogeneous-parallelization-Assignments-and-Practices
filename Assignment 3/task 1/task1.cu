
#include <cuda_runtime.h>                // подключает CUDA runtime
#include <iostream>                  // подключает вывод в консоль
#include <cstdlib>                       // подключает exit()
using namespace std;                               // чтобы не писать std::



void cuda_ok(cudaError_t err, const char* msg) {                // функция проверки ошибок CUDA
    if (err != cudaSuccess) {                                // если произошла ошибка
        cout << "CUDA error (" << msg << "): "                        // печатает место ошибки
             << cudaGetErrorString(err) << endl;                       // печатает текст ошибки
        exit(1);                                       // завершает программу
    }                                             
}                                                

__global__ void kernel_global(int* a, int n, int k) {                     // ядро с global memory
    int i = threadIdx.x + blockIdx.x * blockDim.x;                     // индекс элемента
    if (i < n) {                                           // если индекс в пределах массива
        a[i] = a[i] * k;                                // умножает элемент на k
    }                                                                
}                                                                    
__global__ void kernel_shared(int* a, int n, int k) {                 // ядро с shared memory
    __shared__ int s_data[256];                                 // shared память на блок (под blockDim 256)
    int tid = threadIdx.x;                                         // локальный id потока в блоке
    int i = tid + blockIdx.x * blockDim.x;                       // глобальный индекс элемента

    if (i < n) {                            // если индекс в пределах массива
        s_data[tid] = a[i];                          // копирует элемент из global в shared
    } else {                                  
        s_data[tid] = 0;                         // иначе кладёт 0, чтобы не было мусора
    }                                                                 

    __syncthreads();                // ждет, пока все потоки загрузят shared
    s_data[tid] = s_data[tid] * k;            // умножает значение в shared
    __syncthreads();                                 // ждет, пока все потоки завершат умножение

    if (i < n) {                                       // если индекс в пределах массива
        a[i] = s_data[tid];                         // записывает результат обратно в global
    }                                                               
}                                                                   

float run_kernel_global(int* d_a, int n, int k, int blockSize) {                  // функция запуска global kernel и замера времени
    int gridSize = (n + blockSize - 1) / blockSize;                        // считает количество блоков
    cudaEvent_t start, stop;                                 // создает события CUDA
    cuda_ok(cudaEventCreate(&start), "event create start");              // создает событие start
    cuda_ok(cudaEventCreate(&stop), "event create stop");                 // создает событие stop
    cuda_ok(cudaEventRecord(start), "event record start");                // старт замера времени
    kernel_global<<<gridSize, blockSize>>>(d_a, n, k);                      // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_global launch");                // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");               // стоп замера времени
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");               // ждет завершения
    float ms = 0.0f;                                                    // переменная для миллисекунд
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");            // считает время
    cuda_ok(cudaEventDestroy(start), "event destroy start");              // удаляет событие start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                        // удаляет событие stop
    return ms;                          // возвращает время
}                                                                  

float run_kernel_shared(int* d_a, int n, int k, int blockSize) {                  // функция запуска shared kernel и замера времени
    int gridSize = (n + blockSize - 1) / blockSize;                              // считает количество блоков
    cudaEvent_t start, stop;                                          // создает события CUDA
    cuda_ok(cudaEventCreate(&start), "event create start");                  // создает событие start
    cuda_ok(cudaEventCreate(&stop), "event create stop");                 // создает событие stop
    cuda_ok(cudaEventRecord(start), "event record start");                 // старт замера времени
    kernel_shared<<<gridSize, blockSize>>>(d_a, n, k);                    // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_shared launch");                   // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");                     // стоп замера времени
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");                     // ждет завершения
    float ms = 0.0f;                                                   // переменная для миллисекунд
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");                 // считает время
    cuda_ok(cudaEventDestroy(start), "event destroy start");                         // удаляет событие start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                             // удаляет событие stop
    return ms;                                        // возвращает время
}                                                                 

bool check_result(int* a, int n, int k) {                   // проверка корректности результата
    for (int i = 0; i < n; i++) {                      // проходит по массиву
        int expected = i * k;                           // ожидаемое значение
        if (a[i] != expected) {                        // если значение не совпало
            return false;                                  // возвращает false
        }                                                            
    }                                                              
    return true;                                    // если все совпало
}                                                                    



int main() {                                                          
    const int N = 1000000;                     // размер массива
    const int K = 2;                               // множитель
    const int BLOCK = 256;                 // размер блока 
    int* h_a = new int[N];                    // массив на CPU

    for (int i = 0; i < N; i++) {                   // заполняет массив
        h_a[i] = i;                            // кладет значение i
    }                                                                 

    int* d_a = nullptr;                        // указатель на массив на GPU
    cuda_ok(cudaMalloc((void**)&d_a, N * (int)sizeof(int)), "cudaMalloc d_a");                     // выделяет память на GPU
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy a to GPU");                      // копирует на GPU
    float t_global = run_kernel_global(d_a, N, K, BLOCK);                  // запускает global и меряет время
    cuda_ok(cudaMemcpy(h_a, d_a, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy a to CPU");               // копирует на CPU
    bool ok_global = check_result(h_a, N, K);                                // проверяет корректность global

    for (int i = 0; i < N; i++) {                  // возвращает исходные данные
        h_a[i] = i;                           // снова i
    }                                                                 
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy a to GPU again");                 // снова на GPU
    float t_shared = run_kernel_shared(d_a, N, K, BLOCK);              // запускает shared и меряет время
    cuda_ok(cudaMemcpy(h_a, d_a, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy a to CPU again");                        // копирует на CPU
    bool ok_shared = check_result(h_a, N, K);                            // проверяет корректность shared

    cout << "N = " << N << ", K = " << K << endl;                          // печатает параметры
    cout << "global time: " << t_global << endl;                     // печатает время global
    cout << "shared time: " << t_shared << endl;                         // печатает время shared
    cout << "check global: " << (ok_global ? "ok" : "fail") << endl;                            // печатает проверку global
    cout << "check shared: " << (ok_shared ? "ok" : "fail") << endl;                         // печатает проверку shared

    cuda_ok(cudaFree(d_a), "cudaFree d_a");                // освобождает память на GPU
    delete[] h_a;                                            // освобождает память на CPU

    return 0;               // завершает программу
}                                                          
