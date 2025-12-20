#include <iostream>     // для ввода/вывода
#include <random>       // для генерации случайных чисел
#include <chrono>       // для измерения времени выполнения
#include <omp.h>        // подключение заголовка OpenMP

using namespace std;        // чтобы не писать std:: перед cout/cin/cerr и другими объектами

constexpr int RAND_MIN_VAL = 1;      // минимум диапазона случайных чисел
constexpr int RAND_MAX_VAL = 100;    // максимум диапазона случайных чисел
constexpr size_t PRINT_LIMIT = 20;   // до какого N печатать элементы массива

double averageSequential(const int* arr, size_t N) {
    long long sum = 0;    // переменная sum для накопления суммы всех элементов, long long выбран, чтобы избежать переполнения
    for (size_t i = 0; i < N; ++i) {            // цикл проходит по всем элементам массива (индексы от 0 до N-1)
        sum += arr[i];           // добавляет значение текущего элемента массива к сумме
    }
    return static_cast<double>(sum) / static_cast<double>(N);       // возвращает среднее значение
}

double averageParallel(const int* arr, size_t N) {
    long long sum = 0;     // переменная sum для накопления суммы всех элементов, long long выбран, чтобы избежать переполнения

#pragma omp parallel for reduction(+:sum)       // делит цикл между потоками и складываем частичные суммы в sum
    for (int i = 0; i < static_cast<int>(N); ++i) {     // цикл по массиву (i=int, поэтому N приводим к int)
        sum += arr[i];      // каждый поток добавляет элементы в свою локальную сумму 
    }
    return static_cast<double>(sum) / static_cast<double>(N);       // вычисляем среднее: сумму делим на количество, в типе double
}

int main() {
    cout << "Enter N (array size): ";       // выводит приглашение ввести размер массива
    size_t N;                               // переменная для размера массива
    if (!(cin >> N) || N == 0) {            // проверка ввели ли число и не равен ли размер 0
        cerr << "Invalid size\n";           // если ошибка, то выводит сообщение об ошибке
        return 1;
    }

    int* arr = new int[N];   // динамическое выделение памяти

    random_device rd;
    mt19937 gen(rd());      // генератор случайных чисел
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL);     // распределение: целые числа от RAND_MIN_VAL до RAND_MAX_VAL

    for (size_t i = 0; i < N; ++i) {        // цикл по всем элементам массива
        arr[i] = dist(gen);                 // записывает в arr[i] случайное число из заданного диапазона
    }

    // Печать массива только при небольшом N
    if (N <= PRINT_LIMIT) {
        cout << "Array (first elements): ";     // выводит подпись перед печатью массива
        for (size_t i = 0; i < N; ++i) {        // цикл печати всех элементов массива
            cout << arr[i] << ' ';      // выводит текущий элемент и пробел
        }
        cout << '\n';           // перевод строки после вывода массива
    }
    else {
        cout << "Array size " << N << " (not printed)\n";       // если массив большой не печатает числа
    }

    // Последовательное среднее
    auto t1 = chrono::high_resolution_clock::now();     // запоминает время начала последовательного расчёта
    double avg_seq = averageSequential(arr, N);         // считает среднее значение последовательно
    auto t2 = chrono::high_resolution_clock::now();     // запоминает время окончания последовательного расчёта
    chrono::duration<double, milli> dur_seq = t2 - t1;  // вычисляет длительность (в миллисекундах) для последовательного варианта

    // Параллельное среднее
    auto t3 = chrono::high_resolution_clock::now();     // запоминает время начала параллельного расчёта
    double avg_par = averageParallel(arr, N);           // считает среднее значение параллельно (если OpenMP включён)
    auto t4 = chrono::high_resolution_clock::now();     // запоминает время окончания параллельного расчёта
    chrono::duration<double, milli> dur_par = t4 - t3;  // вычисляет длительность (в миллисекундах) для параллельного варианта 

    cout << "Sequential: average = " << avg_seq             // выводит среднее значение, найденное последовательно
        << ", time = " << dur_seq.count() << " ms\n";       // выводит время последовательного расчёта в миллисекундах
    cout << "Parallel:   average = " << avg_par             // выводит среднее значение, найденное параллельно
        << ", time = " << dur_par.count() << " ms\n";       // выводит время параллельного расчёта в миллисекундах

    if (avg_seq != avg_par) {                                       // проверяет: совпали ли результаты последовательного и параллельного расчёта
        cerr << "sequential and parallel results different!\n";
    }

    cout << "OpenMP threads: " << omp_get_max_threads() << '\n';        // выводит максимальное количество потоков, доступных OpenMP

    delete[] arr;               // освобождает динамический массив, выделенный через new[]
    arr = nullptr;              // обнуляет указатель, чтобы не остался висячий адрес

    return 0;           // завершает программу успешно
}