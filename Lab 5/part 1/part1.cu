
#include <cuda_runtime.h>        // для функций CUDA, которые мне надо
#include <iostream>                  // ввод/вывод
#include <cstdlib>                             // подключает exit()
using namespace std;                           // чтобы не писать std



void cuda_ok(cudaError_t err, const char* msg) {              // функция проверки ошибок CUDA
    if (err != cudaSuccess) {                         // если произошла ошибка
        cout << "CUDA error (" << msg << "): "           // печатает место ошибки
             << cudaGetErrorString(err) << endl;              // печатает текст ошибки
        exit(1);                                           // завершает программу
    }                                                      
}                                                          


struct Stack {                    // структура стека                         
    int* data;                // массив данных стека в GPU памяти                          
    int top;                           // индекс вершины 
    int capacity;                // максимальная емкость стека

    __device__ void init(int* buffer, int size) {                     // инициализация стека на GPU
        data = buffer;                                     // запоминает указатель на буфер
        top = -1;                                      // делает стек пустым 
        capacity = size;                                 // сохраняет ёмкость
    }                                                      

    __device__ bool push(int value) {              // кладёт элемент в стек
        int pos = atomicAdd(&top, 1) + 1;                 // атомарно увеличивает top и получает позицию
        if (pos < capacity) {                  // если место есть
            data[pos] = value;                      // записывает значение в стек
            return true;                     // сообщает успех
        }                                                 
        atomicSub(&top, 1);                         // откатывает top назад, если места нет
        return false;                  // сообщает неуспех
    }                                                 

    __device__ bool pop(int* value) {                  // достаёт элемент из стека
        int pos = atomicSub(&top, 1);                      // атомарно берёт текущий top и уменьшает его
        if (pos >= 0) {                        // если стек не был пустым
            *value = data[pos];                          // записывает извлечённое значение
            return true;                 // сообщает успех
        }                                                  
        atomicAdd(&top, 1);                              // откатывает top назад, если был пустой
        return false;                             // сообщает неуспех
    }                                                     
};                                                        




__global__ void kernel_init(Stack* st, int* buffer, int cap) {                   // ядро для инициализации стека
    if (threadIdx.x == 0 && blockIdx.x == 0) {                 // только один поток выполняет
        st->init(buffer, cap);                        // инициализирует стек
    }                                                          
}                                                           

__global__ void kernel_push(Stack* st, int n_push, int* ok_push) {                // ядро для параллельных push
    int tid = threadIdx.x + blockIdx.x * blockDim.x;                         // вычисляет глобальный id потока
    if (tid < n_push) {                                           // если поток входит в число push
        bool ok = st->push(tid);                             // кладёт tid в стек
        ok_push[tid] = ok ? 1 : 0;                             // сохраняет 1 если push успешен
    }                                                             
}                                                                 

__global__ void kernel_pop(Stack* st, int n_pop, int* out, int* ok_pop) {                    // ядро для параллельных pop
    int tid = threadIdx.x + blockIdx.x * blockDim.x;                                 // вычисляет глобальный id потока
    if (tid < n_pop) {                                                     // если поток входит в число pop
        int val = -1;                                                  // создаёт переменную для значения
        bool ok = st->pop(&val);                               // пытается достать значение из стека
        out[tid] = val;                              // записывает извлечённое значение
        ok_pop[tid] = ok ? 1 : 0;                        // сохраняет 1 если pop успешен
    }                                                                       
}                                                                           

__global__ void kernel_get_top(Stack* st, int* out_top) {                   // ядро чтобы прочитать top
    if (threadIdx.x == 0 && blockIdx.x == 0) {                        // только один поток выполняет
        out_top[0] = st->top;                          // копирует top в массив
    }                                                                       
}                                                                         





int main() {                                                                
    const int CAP = 1024;                     // емкость стека
    const int N_PUSH = 512;                      // сколько элементов кладем
    const int N_POP  = 512;                        // сколько элементов достаем
    int* d_buffer = nullptr;               // буфер данных стека на GPU
    Stack* d_stack = nullptr;                         // объект стека на GPU
    int* d_ok_push = nullptr;                  // массив успешности push на GPU
    int* d_ok_pop  = nullptr;                   // массив извлечённых значений на GPU
    int* d_top_val = nullptr;              // массив для top на GPU
    int* d_out = nullptr;          // массив извлечённых значений на GPU

    cuda_ok(cudaMalloc((void**)&d_buffer, CAP * (int)sizeof(int)), "cudaMalloc d_buffer");                // выделяет буфер стека
    cuda_ok(cudaMalloc((void**)&d_stack, (int)sizeof(Stack)), "cudaMalloc d_stack");                       // выделяет память под Stack
    cuda_ok(cudaMalloc((void**)&d_ok_push, N_PUSH * (int)sizeof(int)), "cudaMalloc d_ok_push");                   // выделяет ok_push
    cuda_ok(cudaMalloc((void**)&d_ok_pop,  N_POP  * (int)sizeof(int)), "cudaMalloc d_ok_pop");                    // выделяет ok_pop
    cuda_ok(cudaMalloc((void**)&d_out,     N_POP  * (int)sizeof(int)), "cudaMalloc d_out");                  // выделяет out
    cuda_ok(cudaMalloc((void**)&d_top_val, (int)sizeof(int)), "cudaMalloc d_top_val");                          // выделяет top_val

    kernel_init<<<1, 1>>>(d_stack, d_buffer, CAP);                     // запускает инициализацию стека
    cuda_ok(cudaGetLastError(), "kernel_init launch");                    // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_init sync");                 // ждет завершения

    int blockSize = 256;                                       // размер блока
    int gridPush = (N_PUSH + blockSize - 1) / blockSize;                       // количество блоков для push
    int gridPop  = (N_POP  + blockSize - 1) / blockSize;                        // количество блоков для pop

    kernel_push<<<gridPush, blockSize>>>(d_stack, N_PUSH, d_ok_push);              // параллельно кладет значения
    cuda_ok(cudaGetLastError(), "kernel_push launch");                      // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_push sync");                      // ждет завершения
    kernel_pop<<<gridPop, blockSize>>>(d_stack, N_POP, d_out, d_ok_pop);                  // параллельно достает значения
    cuda_ok(cudaGetLastError(), "kernel_pop launch");                      // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_pop sync");                            // ждет завершения
    kernel_get_top<<<1, 1>>>(d_stack, d_top_val);                            // читает финальный top
    cuda_ok(cudaGetLastError(), "kernel_get_top launch");                        // проверяет запуск
    cuda_ok(cudaDeviceSynchronize(), "kernel_get_top sync");                       // ждет завершения

    int* h_ok_push = new int[N_PUSH];                         // массив ok_push на CPU
    int* h_ok_pop  = new int[N_POP];                                  // массив ok_pop на CPU
    int* h_out     = new int[N_POP];                            // массив out на CPU
    int  h_top     = 0;                               // переменная top на CPU
    cuda_ok(cudaMemcpy(h_ok_push, d_ok_push, N_PUSH * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy ok_push");                 // копирует ok_push
    cuda_ok(cudaMemcpy(h_ok_pop,  d_ok_pop,  N_POP  * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy ok_pop");                    // копирует ok_pop
    cuda_ok(cudaMemcpy(h_out,     d_out,     N_POP  * (int)sizeof(int), cudaMemcpyDeviceToHost), "copy out");                  // копирует out
    cuda_ok(cudaMemcpy(&h_top,    d_top_val, (int)sizeof(int),           cudaMemcpyDeviceToHost), "copy top");                      // копирует top

    int push_success = 0;                         // счетчик успешных push
    for (int i = 0; i < N_PUSH; i++) {                   // цикл по push
        push_success += h_ok_push[i];                       // суммирует успехи push
    }                                                              
    int pop_success = 0;                              // счетчик успешных pop
    for (int i = 0; i < N_POP; i++) {                       // цикл по pop
        pop_success += h_ok_pop[i];                    // суммирует успехи pop
    }                                                              

    cout << "Push success: " << push_success << " / " << N_PUSH << endl;              // печатает сколько push прошло
    cout << "Pop  success: " << pop_success  << " / " << N_POP  << endl;                // печатает сколько pop прошло
    cout << "Final top value: " << h_top << endl;              // печатает финальный top
    cout << "10 popped values: ";                                     
    for (int i = 0; i < 10 && i < N_POP; i++) {                 // выводит первые 10 значений
        cout << h_out[i] << " ";                                          
    }                                                                      
    cout << endl;                                                           

    cuda_ok(cudaFree(d_buffer), "cudaFree d_buffer");                // освобождает буфер стека
    cuda_ok(cudaFree(d_stack), "cudaFree d_stack");                      // освобождает Stack
    cuda_ok(cudaFree(d_ok_push), "cudaFree d_ok_push");                // освобождает ok_push
    cuda_ok(cudaFree(d_ok_pop), "cudaFree d_ok_pop");                    // освобождает ok_pop
    cuda_ok(cudaFree(d_out), "cudaFree d_out");                        // освобождает out
    cuda_ok(cudaFree(d_top_val), "cudaFree d_top_val");                   // освобождает top_val

    delete[] h_ok_push;                             // освобождает ok_push на CPU
    delete[] h_ok_pop;                                 // освобождает ok_pop на CPU
    delete[] h_out;                               // освобождает out на CPU

    return 0;                  // завершает программу
}                                    
