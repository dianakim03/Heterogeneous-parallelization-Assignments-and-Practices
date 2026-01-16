
#include <cuda_runtime.h>                // подключает CUDA runtime
#include <iostream>                  // подключает вывод в консоль
#include <cstdlib>                       // подключает exit()
using namespace std;                               // чтобы не писать std::


void cuda_ok(cudaError_t err, const char* msg) {     // функция проверки ошибок CUDA
    if (err != cudaSuccess) {     // если произошла ошибка
        cout << "CUDA error (" << msg << "): ";     // печатает место ошибки
        cout << cudaGetErrorString(err) << endl;     // печатает текст ошибки
        exit(1);     // завершает программу
    } 
} 

__global__ void kernel_add(const int* a, const int* b, int* c, int n) {      // CUDA-ядро сложения
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // вычисляет глобальный индекс
    if (i < n) {      // проверяет границы массива
        c[i] = a[i] + b[i];      // складывает элементы
    } 
} 

float run_add(const int* d_a, const int* d_b, int* d_c, int n, int blockSize) {       // функция запуска ядра и замера времени
    int gridSize = (n + blockSize - 1) / blockSize;       // считает количество блоков

    cudaEvent_t start, stop;       // создает события CUDA
    cuda_ok(cudaEventCreate(&start), "event create start");      // создает start
    cuda_ok(cudaEventCreate(&stop), "event create stop");      // создает stop

    cuda_ok(cudaEventRecord(start), "event record start");       // старт таймера
    kernel_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);         // запускает ядро
    cuda_ok(cudaGetLastError(), "kernel_add launch");        // проверяет запуск ядра
    cuda_ok(cudaEventRecord(stop), "event record stop");            // стоп таймера
    cuda_ok(cudaEventSynchronize(stop), "event sync stop");              // ждет завершения ядра

    float ms = 0.0f;                  // переменная для времени
    cuda_ok(cudaEventElapsedTime(&ms, start, stop), "elapsed time");            // получает время в мс

    cuda_ok(cudaEventDestroy(start), "event destroy start");            // удаляет событие start
    cuda_ok(cudaEventDestroy(stop), "event destroy stop");                    // удаляет событие stop

    return ms;                    // возвращает время
}

bool check_sum(const int* a, const int* b, const int* c, int n) {              // проверка корректности
    for (int i = 0; i < n; i++) {                      // проходит по всем элементам
        int expected = a[i] + b[i];                            // считает ожидаемое значение
        if (c[i] != expected) {                     // если значение не совпало
            return false;                        // возвращает false
        } 
    } 
    return true;                 // если все правильно
} 




int main() {
    const int N = 1000000;               // размер массивов
    const int BLOCKS[3] = {128, 256, 512};                 // три размера блока для теста
    int* h_a = new int[N];            // массив a на CPU
    int* h_b = new int[N];            // массив b на CPU
    int* h_c = new int[N];                     // массив c на CPU
    for (int i = 0; i < N; i++) {                  // заполняет массивы
        h_a[i] = i; // a[i] = i
        h_b[i] = 2 * i; // b[i] = 2*i
        h_c[i] = 0; // c[i] = 0
    }
    int* d_a = nullptr;             // указатель a на GPU
    int* d_b = nullptr;                   // указатель b на GPU
    int* d_c = nullptr;                        // указатель c на GPU

    cuda_ok(cudaMalloc((void**)&d_a, N * (int)sizeof(int)), "cudaMalloc d_a");        // выделяет память под a
    cuda_ok(cudaMalloc((void**)&d_b, N * (int)sizeof(int)), "cudaMalloc d_b");                        // выделяет память под b
    cuda_ok(cudaMalloc((void**)&d_c, N * (int)sizeof(int)), "cudaMalloc d_c");               // выделяет память под c
    cuda_ok(cudaMemcpy(d_a, h_a, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy a to GPU");           // копирует a на GPU
    cuda_ok(cudaMemcpy(d_b, h_b, N * (int)sizeof(int), cudaMemcpyHostToDevice), "copy b to GPU");                 // копирует b на GPU

    cout << "N = " << N << endl;               // печатает размер

    for (int t = 0; t < 3; t++) {              // цикл по трем размерам блока
        int blockSize = BLOCKS[t];                            // берет текущий block size
        cuda_ok(cudaMemset(d_c, 0, N * (int)sizeof(int)), "memset d_c");               // обнуляет c на GPU
        float ms = run_add(d_a, d_b, d_c, N, blockSize);                // запускает ядро и меряет время
        cuda_ok(cudaMemcpy(h_c, d_c, N * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy c to CPU");           // копирует результат на CPU
        bool ok = check_sum(h_a, h_b, h_c, N);             // проверяет корректность
        cout << "block size = " << blockSize;                        // печатает block size
        cout << " time = " << ms;                  // печатает время
        cout << " check = " << (ok ? "ok" : "fail") << endl;        // печатает результат проверки
    } 

    cuda_ok(cudaFree(d_a), "cudaFree d_a");                  // освобождает память a на GPU
    cuda_ok(cudaFree(d_b), "cudaFree d_b");                    // освобождает память b на GPU
    cuda_ok(cudaFree(d_c), "cudaFree d_c");                          // освобождает память c на GPU
    delete[] h_a;          // освобождает a на CPU
    delete[] h_b;                // освобождает b на CPU
    delete[] h_c;                   // освобождает c на CPU

    return 0;                          // завершает программу
} 
