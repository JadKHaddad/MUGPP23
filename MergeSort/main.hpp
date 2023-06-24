#pragma once

#include <iostream>
#include <thread>
#include <shared_mutex>
#include <mutex>
#include <memory>

void merge(int *arr, int start, int mid, int end);
void sortAndMerge(int *arr, int start, int mid, int end, int lowerLimit, int freeLogicalCores, std::shared_ptr<std::mutex> mtx);
void mergeSort(int *arr, int start, int end, int lowerLimit, int freeLogicalCores, std::shared_ptr<std::mutex> mtx);
void randomizeArray(int *arr, int n);
void printArray(int *arr, int n);
