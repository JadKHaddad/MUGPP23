#include <iostream>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <memory>

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

void mergeSort(int *arr, int start, int end, int lowerLimit, int freeLogicalCores, std::shared_ptr<std::shared_mutex> mtx)
{
    if (start < end)
    {
        int mid = (start + end) / 2;

        int freeLogicalCores_ = 0;
        {
            std::shared_lock<std::shared_mutex> lock(*mtx);
            freeLogicalCores_ = freeLogicalCores;
        }

        if (freeLogicalCores_ > 0 && end - start > lowerLimit)
        {
            {
                std::unique_lock<std::shared_mutex> lock(*mtx);
                freeLogicalCores--;
            }

            std::thread thread(mergeSort, arr, start, mid, lowerLimit, freeLogicalCores, mtx);
            mergeSort(arr, mid + 1, end, lowerLimit, freeLogicalCores, mtx);
            thread.join();

            {
                std::unique_lock<std::shared_mutex> lock(*mtx);
                freeLogicalCores++;
            }
        }
        else
        {
            mergeSort(arr, start, mid, lowerLimit, freeLogicalCores, mtx);
            mergeSort(arr, mid + 1, end, lowerLimit, freeLogicalCores, mtx);
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
    int freeLogicalCores = std::thread::hardware_concurrency();
    std::shared_ptr<std::shared_mutex> mtx = std::make_shared<std::shared_mutex>();

    int numberOfElements = 1000;
    int lowerLimit = 100;

    int arr[numberOfElements];
    randomize(arr, numberOfElements);
    mergeSort(arr, 0, numberOfElements, lowerLimit, freeLogicalCores, mtx);
    print(arr, numberOfElements);

    return 0;
}