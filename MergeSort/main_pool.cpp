#include <iostream>
#include <vector>
#include <algorithm>
#include <thread>
#include <future>
#include <random>
#include <chrono>
#include <queue>
#include <functional>
#include <mutex>

class ThreadPool
{
public:
    explicit ThreadPool(size_t numThreads)
    {
        for (size_t i = 0; i < numThreads; ++i)
        {
            threads_.emplace_back([this]
                                  {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex_);
                        condition_.wait(lock, [this] {
                            return stop_ || !tasks_.empty();
                        });

                        if (stop_ && tasks_.empty())
                            return;

                        task = std::move(tasks_.front());
                        tasks_.pop();
                    }

                    task();
                } });
        }
    }

    ~ThreadPool()
    {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            stop_ = true;
        }

        condition_.notify_all();

        for (std::thread &thread : threads_)
            thread.join();
    }

    template <class F, class... Args>
    auto enqueue(F &&f, Args &&...args) -> std::future<typename std::result_of<F(Args...)>::type>
    {
        using return_type = typename std::result_of<F(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<return_type> result = task->get_future();
        {
            std::unique_lock<std::mutex> lock(mutex_);
            tasks_.emplace([task]
                           { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

private:
    std::vector<std::thread> threads_;
    std::queue<std::function<void()>> tasks_;

    std::mutex mutex_;
    std::condition_variable condition_;
    bool stop_ = false;
};

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

void sequentialMergeSort(int *arr, int start, int end)
{
    if (start < end)
    {
        int mid = (start + end) / 2;
        sequentialMergeSort(arr, start, mid);
        sequentialMergeSort(arr, mid + 1, end);
        merge(arr, start, mid, end);
    }
}

void parallelMergeSort(int *arr, int start, int end, int lowerLimit, ThreadPool &threadPool)
{
    if (start < end)
    {
        int mid = (start + end) / 2;

        if (end - start > lowerLimit)
        {
            auto leftSort = threadPool.enqueue(parallelMergeSort, arr, start, mid, lowerLimit, std::ref(threadPool));
            parallelMergeSort(arr, mid + 1, end, lowerLimit, threadPool);
            leftSort.wait();
        }
        else
        {
            sequentialMergeSort(arr, start, mid);
            sequentialMergeSort(arr, mid + 1, end);
        }

        merge(arr, start, mid, end);
    }
}

void randomize(int *arr, int n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 1000);

    for (int i = 0; i < n; i++)
        arr[i] = dis(gen);
}

void print(int *arr, int n)
{
    for (int i = 0; i < n; i++)
        std::cout << arr[i] << " ";
    std::cout << std::endl;
}

int main()
{
    const int numElements = 1000;
    const int lowerLimit = 100;

    int arr[numElements];
    randomize(arr, numElements);

    ThreadPool threadPool(std::thread::hardware_concurrency());

    parallelMergeSort(arr, 0, numElements - 1, lowerLimit, threadPool);

    print(arr, numElements);

    return 0;
}
