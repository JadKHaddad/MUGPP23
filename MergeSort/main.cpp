#include "main.hpp"

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

void sortAndMerge(int *arr, int start, int mid, int end, int lowerLimit, int freeLogicalCores, std::shared_ptr<std::mutex> mtx)
{
    mergeSort(arr, start, mid, lowerLimit, freeLogicalCores, mtx);
    mergeSort(arr, mid + 1, end, lowerLimit, freeLogicalCores, mtx);
    merge(arr, start, mid, end);
};

void mergeSort(int *arr, int start, int end, int lowerLimit, int freeLogicalCores, std::shared_ptr<std::mutex> mtx)
{

    if (start < end)
    {
        int mid = (start + end) / 2;

        if (end - start > lowerLimit)
        {
            bool spawnThread = false;
            {
                std::unique_lock<std::mutex> lock(*mtx);
                if (freeLogicalCores > 0)
                {
                    freeLogicalCores--;
                    spawnThread = true;
                }
            }

            if (spawnThread)
            {
                std::thread lovelyThread(mergeSort, arr, start, mid, lowerLimit, freeLogicalCores, mtx);
                mergeSort(arr, mid + 1, end, lowerLimit, freeLogicalCores, mtx);

                lovelyThread.join();
                {
                    std::unique_lock<std::mutex> lock(*mtx);
                    freeLogicalCores++;
                }

                merge(arr, start, mid, end);
                return;
            }

            sortAndMerge(arr, start, mid, end, lowerLimit, freeLogicalCores, mtx);
            return;
        }

        sortAndMerge(arr, start, mid, end, lowerLimit, freeLogicalCores, mtx);
    }
}

void randomizeArray(int *arr, int n)
{
    srand(time(NULL));
    for (int i = 0; i < n; i++)
        arr[i] = rand() % 1000;
}

void printArray(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
{
    int freeLogicalCores = std::thread::hardware_concurrency();
    std::shared_ptr<std::mutex> mtx = std::make_shared<std::mutex>();

    int numberOfElements = 1000;
    int lowerLimit = 100;

    int arr[numberOfElements];
    randomizeArray(arr, numberOfElements);
    mergeSort(arr, 0, numberOfElements, lowerLimit, freeLogicalCores, mtx);
    printArray(arr, numberOfElements);

    return 0;
}
