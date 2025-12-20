#include <iostream>     // для ввода и вывода
#include <omp.h>        // подключение OpenMP
#include <chrono>       // для измерения времени
#include <ctime>        // для time()
#include <cstdlib>      // для rand() и srand()

using namespace std;    // чтобы не писать std:: перед cout


int main() {        // главная функция программы
    const int N = 1000000;           // размер массива 1000000 элементов
    int* arr = new int[N];               // динамически выделяет массив
    srand((unsigned)time(0));            // инициализирует генератор случайных чисел

    for (int i = 0; i < N; i++) {              // цикл для заполнения массива
        arr[i] = rand() % 100 + 1;            // здесь случайные числа от 1 до 100
    }


// ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК
    auto start_sequential = chrono::high_resolution_clock::now();           // время начала поиска
    int min_sequential = arr[0];              // считает первый элемент минимальным
    int max_sequential = arr[0];                 // считает первый элемент максимальным
    for (int i = 1; i < N; i++) {          // в цикле один поток проходится по массиву
        if (arr[i] < min_sequential) min_sequential = arr[i];           // обновляет минимум
        if (arr[i] > max_sequential) max_sequential = arr[i];            // обновляет максимум
    }
    auto end_sequential = chrono::high_resolution_clock::now();   // время окончания поиска
    chrono::duration<double, milli> time_sequential = end_sequential - start_sequential; // длительность поиска
// ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК


// ПАРАЛЛЕЛЬНЫЙ ПОИСК
    auto start_parallel = chrono::high_resolution_clock::now();             // время начала поиска
    int min_parallel = arr[0];            // общий минимум для параллельной версии
    int max_parallel = arr[0];           // общий максимум для параллельной версии

#pragma omp parallel         //  запускает параллельную секцию, где код выполняется одновременно несколькими потоками
    {   int localMin = arr[0];            // локальный минимум потока
        int localMax = arr[0];           // локальный максимум потока

#pragma omp for nowait        // делит цикл по потокам
        for (int i = 1; i < N; i++) {          // цикл, в котором каждый поток обрабатывает часть массива
            if (arr[i] < localMin) localMin = arr[i];       // обновляет локальный минимум
            if (arr[i] > localMax) localMax = arr[i];           // обновляет локальный максимум
        }

#pragma omp critical         // один поток за раз обновляет общий результат
        {   if (localMin < min_parallel) min_parallel = localMin;           // обновляет общий минимум
            if (localMax > max_parallel) max_parallel = localMax;           // обновляет общий максимум
        }
    }
    auto end_parallel = chrono::high_resolution_clock::now();   // время окончания поиска
    chrono::duration<double, milli> time_parallel = end_parallel - start_parallel; // длительность поиска
// ПАРАЛЛЕЛЬНЫЙ ПОИСК
    

    cout << "sequential min: " << min_sequential << endl;    // выводит последовательный min 
    cout << "sequential max: " << max_sequential << endl;    // выводит последовательный max 
    cout << "sequential time: " << time_sequential.count() << endl; // выводит время последовательной версии
    cout << "parallel min: " << min_parallel << endl;      // выводит параллельный min
    cout << "parallel max: " << max_parallel << endl;      // выводит параллельный max
    cout << "parallel time: " << time_parallel.count() << endl;   // выводит время параллельной версии
    cout << "OpenMP threads: " << omp_get_max_threads() << endl; // выводит количество потоков OpenMP

    delete[] arr;                                    // освобождает память
    arr = nullptr;                                   // обнуляет указатель

    return 0;                                        // завершение программы
}