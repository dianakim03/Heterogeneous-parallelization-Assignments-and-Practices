#include <iostream>     // ввод, вывод
#include <vector>       // контейнер vector
#include <random>       // генерация случайных чисел
#include <chrono>        // измерение времени
#include <algorithm>    // sort, is_sorted, inplace_merge
#include <string>        // строки для заголовков
#include <omp.h>        // OpenMP 

using namespace std;  

constexpr int RAND_MIN_VAL = 1;     // минимум диапазона случайных чисел
constexpr int RAND_MAX_VAL = 1000000;      // максимум диапазона случайных чисел
constexpr size_t PRINT_LIMIT = 50;      // печатать массив, только если размер <= PRINT_LIMIT
constexpr bool CHECK_CORRECTNESS = true;    // проверка корректности сортировки

// заполняет массив случайными числами
void fillRandom(vector<int>& a) {                 // функция получает массив по ссылке (меняет его)
    random_device rd;                   // источник случайности от ОС
    mt19937 gen(rd());            // генератор mt19937, "засеянный" rd()
    uniform_int_distribution<int> dist(RAND_MIN_VAL, RAND_MAX_VAL); // распределение целых в диапазоне
    for (auto& x : a) x = dist(gen);          // для каждого элемента массива: записать случайное число
}

// печатает массив (для маленьких N)
void printArray(const vector<int>& a, const string& title) {           // принимает массив (не меняет) и заголовок
    cout << title << ": ";                       // печатает заголовок
    for (auto v : a) cout << v << ' ';       // печатает все элементы через пробел
    cout << '\n';                                    // перевод строки
}

// проверяет как отсортирован массив
bool isSortedCorrect(const vector<int>& a) {         // функция возвращает true/false
    return std::is_sorted(a.begin(), a.end());         // проверка: элементы идут неубывающе
}

// измеряет время выполнения функции сортировки 
template <typename Func>                    // шаблон Func
double measureMs(Func f, vector<int>& a) {             // принимает функцию сортировки и массив
    auto t1 = chrono::high_resolution_clock::now();         // стартовое время
    f(a);           // запускает сортировку (функция меняет массив)
    auto t2 = chrono::high_resolution_clock::now();     // конечное время
    chrono::duration<double, milli> dur = t2 - t1;           // разница во времени в миллисекундах
    return dur.count();         // возвращает число миллисекунд
}



// ЗАДАНИЕ 1. Реализация сортировок без параллелизма (последовательно)
void bubbleSortSeq(vector<int>& a) {         // пузырьковая сортировка (последовательно)
    const size_t N = a.size();          // сохраняет размер массива
    for (size_t i = 0; i < N; ++i) {              // внешний цикл: сколько проходов сделали
        bool swapped = false;                   // флаг: были ли обмены
        for (size_t j = 0; j + 1 < N - i; ++j) {       // проходит по неотсортированной части
            if (a[j] > a[j + 1]) {               // если соседние стоят неправильно
                swap(a[j], a[j + 1]);                  // меняет местами
                swapped = true;                     // запоминает, что был обмен
            }
        }
        if (!swapped) break;         // если обменов не было - массив уже отсортирован
    }
}


void selectionSortSeq(vector<int>& a) {            // сортировка выбором (последовательно)
    const size_t N = a.size();               // размер массива
    for (size_t i = 0; i < N; ++i) {                 // ставиn элемент на позицию i
        size_t minIdx = i;             // индекс минимального элемента
        for (size_t j = i + 1; j < N; ++j) {        // ищет минимум в правой части
            if (a[j] < a[minIdx]) minIdx = j;          // если нашли меньше, то обновляет индекс минимума
        }
        swap(a[i], a[minIdx]);            // ставит минимум в начало текущей части
    }
}


void insertionSortSeq(vector<int>& a) {       // сортировка вставками (последовательно)
    const size_t N = a.size();        // размер массива
    for (size_t i = 1; i < N; ++i) {             // начинаем со 2-го элемента
        int key = a[i];        // текущий элемент, который вставляю
        size_t j = i;                // позиция для сдвига назад
        while (j > 0 && a[j - 1] > key) {           // пока слева элемент больше key
            a[j] = a[j - 1];        // сдвигает элемент вправо
            --j;            // идет левее
        }
        a[j] = key;        // вставляет key на найденную позицию
    }
}



// ЗАДАНИЕ 2. Параллельная реализация сортировок с OpenMP
void bubbleSortPar(vector<int>& a) {        // пузырьковая сортировка параллельно
    const int N = static_cast<int>(a.size());       // размер массива в int 
    for (int phase = 0; phase < N; ++phase) {     // N фаз 
        int start = (phase % 2 == 0) ? 0 : 1;         // чётная фаза: пары (0,1)(2,3), нечётная: (1,2)(3,4)
#pragma omp parallel for               // параллельно делит итерации цикла по потокам
        for (int j = start; j < N - 1; j += 2) {       // шаг 2, чтобы пары не пересекались
            if (a[j] > a[j + 1]) {                   // если пара стоит неправильно
                std::swap(a[j], a[j + 1]);               // меняет элементы в паре
            }
        }
    }
}


void selectionSortPar(vector<int>& a) {        // сортировка выбором параллельно
    const int N = static_cast<int>(a.size());           // размер массива в int
    for (int i = 0; i < N; ++i) {           // внешняя часть идет последовательно
        int globalMinIdx = i;                   // общий индекс минимума
        int globalMinVal = a[i];                    // общее значение минимума

#pragma omp parallel            // создает команду потоков
        {
            int localMinIdx = i;               // локальный индекс минимума у потока
            int localMinVal = a[i];           // локальное значение минимума у потока

#pragma omp for nowait         // делит цикл по потокам, без ожидания в конце
            for (int j = i + 1; j < N; ++j) {         // каждый поток ищет минимум на своём кусочке
                if (a[j] < localMinVal) {         // если нашли меньше локального
                    localMinVal = a[j];            // обновляет локальный минимум (значение)
                    localMinIdx = j;           // обновляет локальный минимум (индекс)
                }
            }

#pragma omp critical               // критическая секция входит только 1 поток одновременно
            {
                if (localMinVal < globalMinVal) {        // если локальный минимум лучше глобального
                    globalMinVal = localMinVal;             // обновляет глобальное значение минимума
                    globalMinIdx = localMinIdx;              // обновляет глобальный индекс минимума
                }
            }
        }
        std::swap(a[i], a[globalMinIdx]);     // ставит найденный минимум в позицию i
    }
}


void insertionSortPar(vector<int>& a) {            // сортировка вставками (параллельная версия через блоки)
    const size_t N = a.size();              // размер массива
    if (N <= 1) return;                         // если 0 или 1 элемент уже отсортировано

    int threads = omp_get_max_threads();       // узнает максимальное число потоков
    if (threads < 1) threads = 1;                 // защита: если вдруг 0, ставим 1

    size_t blockSize = (N + threads - 1) / threads;      // размер блока для каждого потока (округление вверх)

#pragma omp parallel for          // параллельно сортирует каждый блок
    for (int t = 0; t < threads; ++t) {                  // t номер потока
        size_t L = static_cast<size_t>(t) * blockSize;      // левая граница блока
        size_t R = min(N, L + blockSize);          // правая граница блока (не выходит за N)
        if (L >= R) continue;              // если блок пустой пропускает

        for (size_t i = L + 1; i < R; ++i) {       // insertion sort внутри блока
            int key = a[i];       // текущий вставляемый элемент
            size_t j = i;        // позиция для сдвига влево
            while (j > L && a[j - 1] > key) {            // пока не дошли до начала блока и слева больше key
                a[j] = a[j - 1];      // сдвигает вправо
                --j;                          // идет левее
            }
            a[j] = key;         // вставляет key
        }
    }

    // сливает отсортированные блоки
    for (size_t step = blockSize; step < N; step *= 2) {        // step текущий размер уже отсортированных кусков
        for (size_t left = 0; left < N; left += 2 * step) {     // left начало пары кусков
            size_t mid = min(N, left + step);       // середина (конец левого куска)
            size_t right = min(N, left + 2 * step);      // конец правого куска
            if (mid < right) {         // если есть что сливать
                inplace_merge(a.begin() + left,       // слияние двух отсортированных диапазонов
                    a.begin() + mid,
                    a.begin() + right);
            }
        }
    }
}



// ЗАДАНИЕ 3. Сравнение производительности (chrono) и вывод результатов
int main() {           // точка входа программы
    vector<int> sizes = { 1000, 10000, 100000 };         // список размеров для тестов
    cout << "Test sizes: ";       // печатаем "Test sizes:"
    for (size_t i = 0; i < sizes.size(); ++i) {          // выводим каждый размер
        cout << sizes[i] << (i + 1 < sizes.size() ? ", " : "\n"); // запятая или перевод строки
    }

    for (int N : sizes) {        // перебираем каждый размер N
        cout << "Benchmark round N = " << N << "\n";          // печатаем начало теста

        vector<int> base(static_cast<size_t>(N));        // базовый массив размера N
        fillRandom(base);     // заполняем случайными числами

        if (base.size() <= PRINT_LIMIT) {       // если размер маленький
            printArray(base, "Original");           // печатаем массив
        }
        else {                  // иначе не печатаем
            cout << "Original array size " << base.size() << " (not printed)\n"; // сообщение вместо печати
        }

        // правильная сортировка через std::sort (для проверки)
        vector<int> ref = base;            // копия базового массива для эталона
        sort(ref.begin(), ref.end());     // сортируем эталон стандартной сортировкой

        {
            vector<int> seq = base;        // копия для последовательной сортировки
            vector<int> par = base;          // копия для параллельной сортировки

            double tSeq = measureMs(bubbleSortSeq, seq);            // время последовательной версии
            double tPar = measureMs(bubbleSortPar, par);        // время параллельной версии

            cout << "\nBUBBLE\n";           // заголовок Bubble
            cout << "Sequential time: " << tSeq << " ms\n"; // вывод времени seq
            cout << "Parallel   time: " << tPar << " ms\n"; // вывод времени par

            if (CHECK_CORRECTNESS) {        // если включена проверка
                bool okSeq = (seq == ref) && isSortedCorrect(seq);      // seq совпадает с эталоном и отсортирован
                bool okPar = (par == ref) && isSortedCorrect(par);      // par совпадает с эталоном и отсортирован
                cout << "Correctness: seq=" << (okSeq ? "OK" : "FAIL")      // печать результата seq
                    << ", par=" << (okPar ? "OK" : "FAIL") << "\n";           // печать результата par
            }
        }

        {
            vector<int> seq = base;           // копия для последовательной сортировки
            vector<int> par = base;              // копия для параллельной сортировки

            double tSeq = measureMs(selectionSortSeq, seq);         // время seq
            double tPar = measureMs(selectionSortPar, par);      // время par

            cout << "\nSELECTION\n";        // заголовок Selection
            cout << "Sequential time: " << tSeq << " ms\n";         // вывод времени seq
            cout << "Parallel   time: " << tPar << " ms\n";         // вывод времени par

            if (CHECK_CORRECTNESS) {       // если включена проверка
                bool okSeq = (seq == ref) && isSortedCorrect(seq);          // проверка seq
                bool okPar = (par == ref) && isSortedCorrect(par);          // проверка par
                cout << "Correctness: seq=" << (okSeq ? "OK" : "FAIL")      // вывод seq
                    << ", par=" << (okPar ? "OK" : "FAIL") << "\n";           // вывод par
            }
        }

        {
            vector<int> seq = base;           // копия для последовательной сортировки
            vector<int> par = base;                // копия для параллельной сортировки

            double tSeq = measureMs(insertionSortSeq, seq);         // время seq
            double tPar = measureMs(insertionSortPar, par);             // время par

            cout << "\nINSERTION\n";            // заголовок Insertion
            cout << "Sequential time: " << tSeq << " ms\n";             // вывод времени seq
            cout << "Parallel   time: " << tPar << " ms\n";         // вывод времени par

            if (CHECK_CORRECTNESS) {         // если включена проверка
                bool okSeq = (seq == ref) && isSortedCorrect(seq);              // проверка seq
                bool okPar = (par == ref) && isSortedCorrect(par);              // проверка par
                cout << "Correctness: seq=" << (okSeq ? "OK" : "FAIL")      // вывод seq
                    << ", par=" << (okPar ? "OK" : "FAIL") << "\n";     // вывод par
            }
        }
    }
    return 0;      // успешное завершение программы
}