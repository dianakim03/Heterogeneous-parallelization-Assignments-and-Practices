
#include <cuda_runtime.h>               // подключает CUDA runtime
#include <iostream>                // подключает вывод в консоль
#include <cstdlib>                   // подключает exit()
using namespace std;                      // убирает необходимость писать std::


void cuda_ok(cudaError_t err, const char* msg) {                  // объявляет функцию проверки ошибок CUDA
    if (err != cudaSuccess) {                        // проверяет наличие ошибки
        cout << "CUDA error (" << msg << "): ";                 // выводит место ошибки
        cout << cudaGetErrorString(err) << endl;                    // выводит текст ошибки
        exit(1);                   // завершает программу
    } 
} 

__global__ void kernel_add(const int* a, const int* b, int* c, int n) {                 // объявляет ядро сложения
    int i = threadIdx.x + blockIdx.x * blockDim.x;                        // вычисляет глобальный индекс
    if (i < n) {                        // проверяет границы массива
        c[i] = a[i] + b[i];                   // записывает сумму
    } 
} 

float run_add(const int* d_a, const int* d_b, int* d_c, int n, int blockSize) {                 // объявляет функцию запуска и замера времени
    int gridSize = (n + blockSize - 1) / blockSize;                    // вычисляет количество блоков
    cudaEvent_t start, stop;                                  // объявляет CUDA-события
    cuda_ok(cudaEventCreate(&start), "event create start");                    // создаёт start
    cuda_ok(cudaEventCreate(&stop), "event create stop");                          // создаёт stop
    cuda_ok(cudaEventRecord(start), "event record start");                    // запускает таймер
    kernel_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);                       // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_add launch");                    // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");                  // останавливает таймер
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");                       // ожидает завершения ядра
    float ms = 0.0f;                                                 // объявляет переменную времени
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");                   // получает время
    cuda_ok(cudaEventDestroy(start), "event destroy start");              // удаляет start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                       // удаляет stop

    return ms;                  // возвращает время
} 

bool check_sum(const int* a, const int* b, const int* c, int n) {                  // объявляет функцию проверки
    for (int i = 0; i < n; i++) {                         // проходит по массиву
        int expected = a[i] + b[i];                             // вычисляет ожидаемое значение
        if (c[i] != expected) {                                   // проверяет совпадение
            return false;                             // возвращает false при ошибке
        }
    } 
    return true;                        // возвращает true при успехе
}
 


int main() { 
    const int N = 1000000;        // задаёт размер массивов
    const int BLOCK_BAD = 32;               // задаёт неоптимальный block size
    const int BLOCK_GOOD = 256;        // задаёт более оптимальный block size
    int* h_a = new int[N];                    // выделяет массив a на CPU
    int* h_b = new int[N];               // выделяет массив b на CPU
    int* h_c = new int[N];                // выделяет массив c на CPU
    for (int i = 0; i < N; i++) {                 // заполняет массивы
        h_a[i] = i;                            // задаёт a[i]
        h_b[i] = 2 * i;                             // задаёт b[i]
        h_c[i] = 0;                            // обнуляет c[i]
    } 
    int* d_a = nullptr;         // объявляет указатель a на GPU
    int* d_b = nullptr;                          // объявляет указатель b на GPU
    int* d_c = nullptr;                    // объявляет указатель c на GPU

    cuda_ok(cudaMalloc((void**)&d_a, N * (int)sizeof(int)), "cudaMalloc d_a");            // выделяет память под a
    cuda_ok(cudaMalloc((void**)&d_b, N * (int)sizeof(int)), "cudaMalloc d_b");                  // выделяет память под b
    cuda_ok(cudaMalloc((void**)&d_c, N * (int)sizeof(int)), "cudaMalloc d_c");                        // выделяет память под c
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy a to GPU");          // копирует a на GPU
    cuda_ok(cudaMemcpy(d_b, h_b, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy b to GPU");              // копирует b на GPU
    cuda_ok(cudaMemset(d_c, 0, N * (int)sizeof(int)), "memset d_c");                // обнуляет c на GPU
    float t_bad = run_add(d_a, d_b, d_c, N, BLOCK_BAD);                      // измеряет время неоптимального варианта
    cuda_ok(cudaMemcpy(h_c, d_c, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy c bad");                // копирует результат
    bool ok_bad = check_sum(h_a, h_b, h_c, N);                                   // проверяет корректность
    cuda_ok(cudaMemset(d_c, 0, N * (int)sizeof(int)), "memset d_c");                            // обнуляет c на GPU
    float t_good = run_add(d_a, d_b, d_c, N, BLOCK_GOOD);                      // измеряет время оптимального варианта
    cuda_ok(cudaMemcpy(h_c, d_c, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy c good");                     // копирует результат
    bool ok_good = check_sum(h_a, h_b, h_c, N);                                        // проверяет корректность

    cout << "N = " << N << endl;                       // выводит N
    cout << "bad block size = " << BLOCK_BAD << endl;                        // выводит плохой block size
    cout << "good block size = " << BLOCK_GOOD << endl;                   // выводит хороший block size
    cout << "bad config time: " << t_bad << "check = " << (ok_bad ? "ok" : "fail") << endl;                  // выводит время и проверку
    cout << "good config time: " << t_good << "check = " << (ok_good ? "ok" : "fail") << endl;                  // выводит время и проверку

    cuda_ok(cudaFree(d_a), "cudaFree d_a");            // освобождает d_a
    cuda_ok(cudaFree(d_b), "cudaFree d_b");                // освобождает d_b
    cuda_ok(cudaFree(d_c), "cudaFree d_c");                           // освобождает d_c
    delete[] h_a;                     // освобождает h_a
    delete[] h_b;                            // освобождает h_b
    delete[] h_c;                             // освобождает h_c

    return 0;            // завершает программу
} 
