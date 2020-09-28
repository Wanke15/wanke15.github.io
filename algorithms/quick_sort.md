```python
# arr = [10, 7, 8, 9, 1, 5]


def partition(arr, low, high):
    i = low - 1  # 最小元素索引
    pivot = arr[high]

    for j in range(low, high):
        # 当前元素小于或等于 pivot
        # 按从左往右的顺序依次把小于pivot的值挪到左边
        if arr[j] <= pivot:
            i = i + 1
            arr[i], arr[j] = arr[j], arr[i]
    
    # 最后交换pivot
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    # 返回pivot的位置
    return i + 1


# arr[] --> 排序数组
# low  --> 起始索引
# high  --> 结束索引

# 快速排序函数
def quickSort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)

        quickSort(arr, low, pi - 1)
        quickSort(arr, pi + 1, high)


arr = [10, 7, 8, 9, 1, 5]
print(arr.sort())
n = len(arr)
quickSort(arr, 0, n - 1)
print("排序后的数组:")
for i in range(n):
    print("%d" % arr[i])

```
