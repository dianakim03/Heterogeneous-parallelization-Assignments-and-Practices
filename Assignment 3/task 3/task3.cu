
#include <cuda_runtime.h>                 // подключаем CUDA runtime
#include <iostream>                             // для вывода
#include <cstdlib>                       // для exit()
using namespace std;                         // чтобы не писать std::


void cuda_ok(cudaError_t err, const char* msg) {              // объявляет функцию проверки ошибок CUDA
    if (err != cudaSuccess) {                       // проверяет наличие ошибки
        cout << "CUDA error (" << msg << "): ";            // выводит место ошибки
        cout << cudaGetErrorString(err) << endl;               // выводит текст ошибки
        exit(1);              // завершает программу
    } 
} 

__global__ void kernel_coalesced(int* a, int n) {           // объявляет ядро с коалесцированным доступом
    int i = threadIdx.x + blockIdx.x * blockDim.x;                       // вычисляет глобальный индекс
    if (i < n) {                  // проверяет выход за границы массива
        a[i] = a[i] + 1;                    // увеличивает элемент на 1
    } 
} 

__global__ void kernel_noncoalesced(int* a, int n, int stride) {              // объявляет ядро с некоалесцированным доступом
    int i = threadIdx.x + blockIdx.x * blockDim.x;                            // вычисляет глобальный индекс
    if (i < n) {                // проверяет границы массива
        int idx = (i * stride) % n;                // вычисляет индекс со stride
        a[idx] = a[idx] + 1;             // увеличивает элемент по скачущему индексу
    } 
}

float run_coalesced(int* d_a, int n, int blockSize) {                   // объявляет функцию запуска coalesced ядра
    int gridSize = (n + blockSize - 1) / blockSize;                // вычисляет количество блоков
    cudaEvent_t start, stop;                 // объявляет CUDA-события
    cuda_ok(cudaEventCreate(&start), "event create start");               // создает start
    cuda_ok(cudaEventCreate(&stop), "event create stop");            // создаёт stop
    cuda_ok(cudaEventRecord(start), "event record start");       // запускает таймер
    kernel_coalesced<<<gridSize, blockSize>>>(d_a, n);                  // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_coalesced launch");                         // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");                     // останавливает таймер
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");            // ожидает завершения ядра
    float ms = 0.0f;             // объявляет переменную для времени
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");                 // получает время в мс
    cuda_ok(cudaEventDestroy(start), "event destroy start");        // удаляет start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                             // удаляет stop
    return ms;            // возвращает время
}

float run_noncoalesced(int* d_a, int n, int blockSize, int stride) {                     // объявляет функцию запуска noncoalesced ядра
    int gridSize = (n + blockSize - 1) / blockSize;                              // вычисляет количество блоков
    cudaEvent_t start, stop;                    // объявляет CUDA-события
    cuda_ok(cudaEventCreate(&start), "event create start");                      // создает start
    cuda_ok(cudaEventCreate(&stop), "event create stop");               // создает stop
    cuda_ok(cudaEventRecord(start), "event record start");                // запускает таймер
    kernel_noncoalesced<<<gridSize, blockSize>>>(d_a, n, stride);                    // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_noncoalesced launch");               // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");                        // останавливает таймер
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");                  // ожидает завершения ядра
    float ms = 0.0f;                                         // объявляет переменную для времени
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");                       // получает время в мс
    cuda_ok(cudaEventDestroy(start), "event destroy start");      // удаляет start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                  // удаляет stop
    return ms;          // возвращает время
} 

bool check_plus_one(const int* a, int n) {                // объявляет функцию проверки результата
    for (int i = 0; i < n; i++) {               // проходит по массиву
        if (a[i] != i + 1) {                  // проверяет ожидаемое значение
            return false;                           // возвращает false при ошибке
        } 
    } 
    return true;             // возвращает true если всё правильно
} 




int main() { 
    const int N = 1000000;          // задает размер массива
    const int BLOCK = 256;                // задает размер блока
    const int STRIDE = 33;                // задает stride для плохого доступа
    int* h_a = new int[N];                    // выделяет массив на CPU
    for (int i = 0; i < N; i++) {               // заполняет массив
        h_a[i] = i;                   // записывает i
    }
    int* d_a = nullptr;                     // объявляет указатель на GPU
    cuda_ok(cudaMalloc((void**)&d_a, N * (int)sizeof(int)), "cudaMalloc d_a");          // выделяет память на GPU
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy to GPU");               // копирует на GPU
    float t_coal = run_coalesced(d_a, N, BLOCK);                  // запускает coalesced и измеряет время
    cuda_ok(cudaMemcpy(h_a, d_a, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy back coal");              // копирует результат
    bool ok_coal = check_plus_one(h_a, N);                            // проверяет результат coalesced
    for (int i = 0; i < N; i++) {                         // восстанавливает исходные значения
        h_a[i] = i;                           // снова записывает i
    } 
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy to GPU again");                 // копирует снова
    float t_non = run_noncoalesced(d_a, N, BLOCK, STRIDE);                            // запускает noncoalesced и измеряет время
    cuda_ok(cudaMemcpy(h_a, d_a, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy back non");                // копирует результат
    bool ok_non = check_plus_one(h_a, N);                                       // проверяет результат noncoalesced

    cout << "N = " << N << endl;                // выводит N
    cout << "block = " << BLOCK << endl;            // выводит block
    cout << "stride = " << STRIDE << endl;              // выводит stride
    cout << "coalesced time: " << t_coal << endl;              // выводит время coalesced
    cout << "non-coalesced time: " << t_non << endl;                  // выводит время noncoalesced
    cout << "check coalesced: " << (ok_coal ? "ok" : "fail") << endl;           // выводит проверку coalesced
    cout << "check non-coalesced: " << (ok_non ? "ok" : "fail") << endl;               // выводит проверку noncoalesced

    cuda_ok(cudaFree(d_a), "cudaFree d_a");             // освобождает GPU память
    delete[] h_a;                    // освобождает CPU память

    return 0;          // завершает программу
} 
