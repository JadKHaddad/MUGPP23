#include <iostream>
#include <thread>
#include <shared_mutex>
#include <mutex>

std::mutex mutex;
int freeLogicalCores = 0;

void merge(int *arr, int start, int mid, int end)
{
    int *temp = new int[end - start + 1];
    int i = start, j = mid + 1, k = 0;

    while (i <= mid && j <= end)
    {
        if (arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while (i <= mid)
        temp[k++] = arr[i++];

    while (j <= end)
        temp[k++] = arr[j++];

    for (i = start; i <= end; i++)
        arr[i] = temp[i - start];

    delete[] temp;
}

void parallelMergeSort(int *arr, int start, int end, int lowerLimit)
{
    if (start < end)
    {
        int mid = (start + end) / 2;

        int freeLogicalCoresCount = 0;
        mutex.lock();
        freeLogicalCoresCount = freeLogicalCores;
        mutex.unlock();

        if (freeLogicalCoresCount > 0 && end - start > lowerLimit)
        {
            mutex.lock();
            freeLogicalCores--;
            mutex.unlock();

            std::thread t(parallelMergeSort, arr, start, mid, lowerLimit);
            parallelMergeSort(arr, mid + 1, end, lowerLimit);
            t.join();

            mutex.lock();
            freeLogicalCores++;
            mutex.unlock();
        }
        else
        {
            parallelMergeSort(arr, start, mid, lowerLimit);
            parallelMergeSort(arr, mid + 1, end, lowerLimit);
        }
        merge(arr, start, mid, end);
    }
}

void randomize(int *arr, int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;
}

void print(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
{
    freeLogicalCores = std::thread::hardware_concurrency();

    int numberOfElements = 100;
    int lowerLimit = 10;

    int arr[numberOfElements];
    randomize(arr, numberOfElements);
    parallelMergeSort(arr, 0, numberOfElements, lowerLimit);
    print(arr, numberOfElements);

    return 0;
}