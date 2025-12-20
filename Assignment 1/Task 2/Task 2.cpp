#include <iostream>     // для ввода и вывода
#include <chrono>       // для измерения времени
#include <ctime>        // для time()
#include <cstdlib>      // для rand() и srand()

using namespace std;    // чтобы не писать std:: перед cout


int main() {          // главная функция программы
    srand(time(0));                  // инициализирует генератор случайных чисел
    const int N = 1000000;       // размер массива 1000000 элементов
    int* arr = new int[N];            // динамически выделяет массив из N целых чисел

    for (int i = 0; i < N; i++) {          // цикл проходитя по всем элементам массива
        arr[i] = rand() % 100 + 1;            // заполняет массив случайными числами от 1 до 100
    }

    auto start = chrono::high_resolution_clock::now();      // запоминает время начала алгоритма
    int minVal = arr[0];             // считает первый элемент минимальным
    int maxVal = arr[0];           // считает первый элемент максимальным

    for (int i = 1; i < N; i++) {         // цикл для последовательного поиска мин. и макс. значения
        if (arr[i] < minVal) {                // если текущий элемент меньше текущего минимума
            minVal = arr[i];                    // то он обновляет минимум
        }
        if (arr[i] > maxVal) {         // если текущий элемент больше текущего максимума
            maxVal = arr[i];               // то он обновляет максимум
        }
    }

    auto end = chrono::high_resolution_clock::now();        // запоминает время окончания алгоритма
    chrono::duration<double, milli> duration = end - start;             // вычисляет время выполнения в миллисекундах

    cout << "minimum: " << minVal << endl;        // выводит минимальное значение
    cout << "maximum: " << maxVal << endl;          // выводит максимальное значение
    cout << "time: " << duration.count() << endl;          // выводит время выполнения алгоритма

    delete[] arr;                               // освобождает динамически выделенную память
    arr = nullptr;                              // обнуляет указатель

    return 0;                                   // завершение программы
}