
#include <cuda_runtime.h>        // для функций CUDA, которые мне надо
#include <iostream>                  // ввод/вывод
#include <cstdlib>                             // подключает exit()
using namespace std;                           // чтобы не писать std



void cuda_ok(cudaError_t err, const char* msg) {           // функция проверки ошибок CUDA
    if (err != cudaSuccess) {                                 // если произошла ошибка
        cout << "CUDA error (" << msg << "): "            // печатает место ошибки
             << cudaGetErrorString(err) << endl;              // печатает текст ошибки
        exit(1);                                // завершает программу
    }                                                  
}                                                         

struct Queue {                                 // структура очереди
    int* data;                                    // массив данных очереди в GPU памяти
    int head;                           // индекс головы, то еть откуда читаем
    int tail;                                 // индекс хвоста, то еть куда пишем
    int capacity;                                // максимальная емкость очереди

    __device__ void init(int* buffer, int size) {               // инициализация очереди на GPU
        data = buffer;                                // запоминает указатель на буфер
        head = 0;                                 // голова начинается с 0
        tail = 0;                              // хвост начинается с 0
        capacity = size;                            // сохраняет емкость
    }                                                     
    __device__ bool enqueue(int value) {                 // добавляет элемент в очередь
        int pos = atomicAdd(&tail, 1);              // атомарно берет позицию и увеличивает tail
        if (pos < capacity) {                 // если место в очереди есть
            data[pos] = value;                 // записывает значение в очередь
            return true;                          // сообщает успех
        }                                                  
        return false;                          // сообщает неуспех, очередь переполнена
    }                                                     
    __device__ bool dequeue(int* value) {                  // удаляет элемент из очереди
        int pos = atomicAdd(&head, 1);                      // атомарно берёт позицию и увеличивает head
        if (pos < tail) {                              // если элемент реально существует
            *value = data[pos];                          // записывает извлечённое значение
            return true;                           // сообщает успех
        }                                                 
        return false;                                // сообщает неуспех, очередь пуста
    }                                                      
};                                                         



__global__ void kernel_init(Queue* q, int* buffer, int cap) {           // ядро инициализации очереди
    if (threadIdx.x == 0 && blockIdx.x == 0) {                   // только один поток выполняет
        q->init(buffer, cap);                              // инициализирует очередь
    }                                                      
}                                                              
__global__ void kernel_enqueue(Queue* q, int n_enq, int* ok_enq) {       // ядро для параллельных enqueue
    int tid = threadIdx.x + blockIdx.x * blockDim.x;                  // вычисляет глобальный id потока
    if (tid < n_enq) {                                  // если поток входит в число enqueue
        bool ok = q->enqueue(tid);                             // добавляет tid в очередь
        ok_enq[tid] = ok ? 1 : 0;                               // пишет 1 если успешно
    }                                                              
}                                                                  
__global__ void kernel_dequeue(Queue* q, int n_deq, int* out, int* ok_deq) {                 // ядро для параллельных dequeue
    int tid = threadIdx.x + blockIdx.x * blockDim.x;                             // вычисляет глобальный id потока
    if (tid < n_deq) {                                                             // если поток входит в число dequeue
        int val = -1;                                              // значение по умолчанию
        bool ok = q->dequeue(&val);                                     // пытается достать значение
        out[tid] = val;                                              // сохраняет значение
        ok_deq[tid] = ok ? 1 : 0;                                    // пишет 1 если успешно
    }                                                                             
}                                                                                 
__global__ void kernel_get_head_tail(Queue* q, int* out_head, int* out_tail) {           // ядро чтения head и tail
    if (threadIdx.x == 0 && blockIdx.x == 0) {                            // только один поток выполняет
        out_head[0] = q->head;                                          // копирует head
        out_tail[0] = q->tail;                                            // копирует tail
    }                                                                             
}                                                                                 




int main() {                                                                       
    const int CAP = 1024;                          // емкость очереди
    const int N_ENQ = 512;                              // сколько добавляем
    const int N_DEQ = 512;                       // сколько извлекаем
    int* d_buffer = nullptr;                           // буфер очереди на GPU
    Queue* d_queue = nullptr;                   // объект очереди на GPU
    int* d_ok_enq = nullptr;                   // массив успешности enqueue на GPU
    int* d_ok_deq = nullptr;                     // массив успешности dequeue на GPU
    int* d_out = nullptr;                 // массив извлеченных значений на GPU
    int* d_head_val = nullptr;                            // массив для head на GPU
    int* d_tail_val = nullptr;                          // массив для tail на GPU

    cuda_ok(cudaMalloc((void**)&d_buffer, CAP * (int)sizeof(int)), "cudaMalloc d_buffer");               // выделяет буфер очереди
    cuda_ok(cudaMalloc((void**)&d_queue, (int)sizeof(Queue)), "cudaMalloc d_queue");                  // выделяет память под Queue
    cuda_ok(cudaMalloc((void**)&d_ok_enq, N_ENQ * (int)sizeof(int)), "cudaMalloc d_ok_enq");                 // выделяет ok_enq
    cuda_ok(cudaMalloc((void**)&d_ok_deq, N_DEQ * (int)sizeof(int)), "cudaMalloc d_ok_deq");                       // выделяет ok_deq
    cuda_ok(cudaMalloc((void**)&d_out, N_DEQ * (int)sizeof(int)), "cudaMalloc d_out");                         // выделяет out
    cuda_ok(cudaMalloc((void**)&d_head_val, (int)sizeof(int)), "cudaMalloc d_head_val");                          // выделяет head_val
    cuda_ok(cudaMalloc((void**)&d_tail_val, (int)sizeof(int)), "cudaMalloc d_tail_val");                         // выделяет tail_val

    kernel_init<<<1, 1>>>(d_queue, d_buffer, CAP);                            // запускает инициализацию очереди
    cuda_ok(cudaGetLastError(), "kernel_init launch");                           // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_init sync");                // ждет завершения
    int blockSize = 256;                                               // размер блока
    int gridEnq = (N_ENQ + blockSize - 1) / blockSize;                       // количество блоков для enqueue
    int gridDeq = (N_DEQ + blockSize - 1) / blockSize;                             // количество блоков для dequeue
    kernel_enqueue<<<gridEnq, blockSize>>>(d_queue, N_ENQ, d_ok_enq);                         // параллельно добавляет значения
    cuda_ok(cudaGetLastError(), "kernel_enqueue launch");                                          // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_enqueue sync");                                      // ждет завершения
    kernel_dequeue<<<gridDeq, blockSize>>>(d_queue, N_DEQ, d_out, d_ok_deq);                        // параллельно извлекает значения
    cuda_ok(cudaGetLastError(), "kernel_dequeue launch");                                      // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_dequeue sync");                                     // ждет завершения
    kernel_get_head_tail<<<1, 1>>>(d_queue, d_head_val, d_tail_val);                          // читает head и tail
    cuda_ok(cudaGetLastError(), "kernel_get_head_tail launch");                                // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_get_head_tail sync");                         // ждет завершения

    int* h_ok_enq = new int[N_ENQ];                   // ok_enq на CPU
    int* h_ok_deq = new int[N_DEQ];               // ok_deq на CPU
    int* h_out = new int[N_DEQ];                    // out на CPU
    int h_head = 0;                            // head на CPU
    int h_tail = 0;                               // tail на CPU
    cuda_ok(cudaMemcpy(h_ok_enq, d_ok_enq, N_ENQ * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy ok_enq");                 // копирует ok_enq
    cuda_ok(cudaMemcpy(h_ok_deq, d_ok_deq, N_DEQ * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy ok_deq");                 // копирует ok_deq
    cuda_ok(cudaMemcpy(h_out, d_out, N_DEQ * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy out");                     // копирует out
    cuda_ok(cudaMemcpy(&h_head, d_head_val, (int)sizeof(int), cudaMemcpyDeviceToHost), "copy head");                       // копирует head
    cuda_ok(cudaMemcpy(&h_tail, d_tail_val, (int)sizeof(int), cudaMemcpyDeviceToHost), "copy tail");                   // копирует tail

    int enq_success = 0;                           // счетчик успешных enqueue
    for (int i = 0; i < N_ENQ; i++) {                  // цикл по enqueue
        enq_success += h_ok_enq[i];                // суммирует успехи enqueue
    }                                                                                                
    int deq_success = 0;                                // счетчик успешных dequeue
    for (int i = 0; i < N_DEQ; i++) {               // цикл по dequeue
        deq_success += h_ok_deq[i];                   // суммирует успехи dequeue
    }                                                                                                

    cout << "enqueue success: " << enq_success << " / " << N_ENQ << endl;                    // печатает успех enqueue
    cout << "dequeue success: " << deq_success << " / " << N_DEQ << endl;                       // печатает успех dequeue
    cout << "final head: " << h_head << ", final tail: " << h_tail << endl;                    // печатает head и tail
    cout << "first 10 dequeued values: ";                                                       
    for (int i = 0; i < 10 && i < N_DEQ; i++) {                                     // выводит первые 10 значений
        cout << h_out[i] << " ";                                              
    }                                                                                               
    cout << endl;                                                                                  
    
    cuda_ok(cudaFree(d_buffer), "cudaFree d_buffer");           // освобождает буфер
    cuda_ok(cudaFree(d_queue), "cudaFree d_queue");                   // освобождает объект очереди
    cuda_ok(cudaFree(d_ok_enq), "cudaFree d_ok_enq");                      // освобождает ok_enq
    cuda_ok(cudaFree(d_ok_deq), "cudaFree d_ok_deq");                       // освобождает ok_deq
    cuda_ok(cudaFree(d_out), "cudaFree d_out");                         // освобождает out
    cuda_ok(cudaFree(d_head_val), "cudaFree d_head_val");                 // освобождает head_val
    cuda_ok(cudaFree(d_tail_val), "cudaFree d_tail_val");                  // освобождает tail_val

    delete[] h_ok_enq;                            // освобождает ok_enq на CPU
    delete[] h_ok_deq;                             // освобождает ok_deq на CPU
    delete[] h_out;                         // освобождает out на CPU

    return 0;        // завершает программу
}  
