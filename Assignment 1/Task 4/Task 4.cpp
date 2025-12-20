#include <iostream>     // для ввода и вывода
#include <omp.h>        // подключение OpenMP
#include <chrono>       // для измерения времени
#include <ctime>        // для time()
#include <cstdlib>      // для rand() и srand()

using namespace std;    // чтобы не писать std:: перед cout


int main() {        // главная функция программы
    const int N = 5000000;          // размер массива 5000000 элементов
    int* arr = new int[N];               // динамически выделяет память под массив
    srand((unsigned)time(0));        // инициализирует генератор случайных чисел
    for (int i = 0; i < N; i++) {         // цикл заполняет массив
        arr[i] = rand() % 100 + 1;               // случайными числами от 1 до 100
    }


// ПОСЛЕДОВАТЕЛЬНЫЙ ПОДСЧЕТ
    auto start_sequential = chrono::high_resolution_clock::now();       // время начала подсчета
    long long sum_sequential = 0;       // сумма элементов
    for (int i = 0; i < N; i++) {       // цикл проходится по всем элементам массива
        sum_sequential += arr[i];            // прибавляет текущий элемент к сумме
    }
    double avg_sequential = (double)sum_sequential / N;           // вычисляет среднее значение
    auto end_sequential = chrono::high_resolution_clock::now();   // время окончания подсчета
    chrono::duration<double, milli> time_sequential = end_sequential - start_sequential; // длительность подсчёта
// ПОСЛЕДОВАТЕЛЬНЫЙ ПОДСЧЕТ



// ПАРАЛЛЕЛЬНЫЙ ПОДСЧЕТ
    auto start_parallel = chrono::high_resolution_clock::now();          // время начала подсчета
    long long sum_parallel = 0;          // сумма элементов

#pragma omp parallel for reduction(+:sum_parallel)       // параллельный цикл, плюс редукция суммы
    for (int i = 0; i < N; i++) {       // цикл делится между потоками
        sum_parallel += arr[i];           // и каждый поток считает свою часть суммы
    }
    double avg_parallel = (double)sum_parallel / N;           // вычисляет среднее значение
    auto end_parallel = chrono::high_resolution_clock::now();    // время окончания подсчета
    chrono::duration<double, milli> time_parallel = end_parallel - start_parallel;    // длительность подсчета
// ПАРАЛЛЕЛЬНЫЙ ПОДСЧЕТ


    cout << "sequential average: " << avg_sequential << endl;       // вывод последовательного среднего
    cout << "sequential time: " << time_sequential.count() << endl;      // вывод последовательного времени
    cout << "parallel average: " << avg_parallel << endl;       // вывод параллельного среднего
    cout << "parallel time: " << time_parallel.count() << endl;     // вывод параллельного времени
    cout << "OpenMP threads: " << omp_get_max_threads() << endl;        // вывод количества потоков

    delete[] arr;        // освобождает память
    arr = nullptr;         // обнуляет указатель

    return 0;     // завершение программы
}