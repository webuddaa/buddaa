#--------------------------------------

def indexOfMin(l):
    '''
    :param l: 一个列表
    :return: 列表中最小数的索引
    '''
    if len(l) == 0:
        return None
    if len(l) == 1:
        return 0

    minIndex = 0
    currentIndex = 1
    while currentIndex < len(l):
        if l[minIndex] > l[currentIndex]:
            minIndex = currentIndex
        currentIndex += 1
    return minIndex

#----------------------------

def linearSearch(target, l):
    '''
    顺序搜索列表中的目标项
    :param target: 目标项
    :param l: 一个列表
    :return: 目标项的索引，如果没有，则返回-1
    '''
    currentIndex = 0
    while currentIndex < len(l):
        if target == l[currentIndex]:
            return currentIndex
        else:
            currentIndex += 1
    return -1

#-----------------------------------

def binarySearch(target, sortedList):
    '''
    二分查找法
    :param target:
    :param sortedList:
    :return: 目标项的索引
    '''
    left = 0
    right = len(sortedList) - 1
    while left <= right:
        mid = (left + right) // 2
        if target == sortedList[mid]:
            return mid
        elif target < sortedList[mid]:
            right = mid - 1
        else:
            left = mid + 1
    return -1

#-----------------------------------

def interpolationSearch(target, sortedList):
    '''
    插值查找法，适用于分布均匀的查找表，整体上与二分查找很像
    :param target:
    :param sortedList:
    :return:目标项的索引
    '''
    left = 0
    right = len(sortedList) - 1
    while left <= right:
        # 下面一行就是与二分查找唯一的区别了
        mid = left + (right - left) * (target - sortedList[left]) / (sortedList[right] - sortedList[left])
        mid = int(mid)
        print("the left: %d, the right: %d, the mid: %d" % (left, right, mid))
        if target == sortedList[mid]:
            return mid
        elif target < sortedList[mid]:
            right = mid - 1
        else:
            left = mid + 1

    return -1

#-------------------------------------

def fibonacci(x):
    if x <= 0:
        return 0
    a, b = 1, 1
    for _ in range(x-1):
        a, b = b, a+b
    return a
def fibonacciSearch(target, sortedList):
    # 生成fibonacci数列
    l = []
    for i in range(1, 20):
        l.append(fibonacci(i))

    length = len(sortedList)
    low, high = 0, length-1
    k = 0
    while length > l[k]-1:
        k += 1

    for i in range(length, l[k]-1):
        l.append(sortedList[length-1])

    while low <= high:
        mid = low + l[k-1] - 1
        if target > sortedList[mid]:
            low = mid + 1
            k -= 2
        elif target < sortedList[mid]:
            high = mid - 1
            k -= 1
        else:
            if mid <= length-1:
                return mid
            else:
                return length-1

    return -1

#-------------------------------------

def swap(l, i, j):
    '''
    在列表中交换两个元素的位置
    :param l: 一个列表
    :param i: 被交换的元素索引
    :param j: 被交换的元素索引
    :return:
    '''
    # temp = l[i]
    # l[i] = l[j]
    # l[j] = temp
    l[i], l[j] = l[j], l[i]

#-------------------------------------

def selectionSort(l):
    '''
    选择排序法，时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    i = 0
    length = len(l)
    while i < length - 1:
        minIndex = i
        j = i + 1
        while j < length:
            if l[j] < l[minIndex]:
                minIndex = j
            j += 1
        if minIndex != i:
            swap(l, minIndex, i)
        i += 1

#-------------------------------------

def bubbleSort(l):
    '''
    冒泡排序法, 时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    for i in range(len(l)):
        for j in range(i + 1, len(l)):
            if l[i] > l[j]:
                l[i], l[j] = l[j], l[i]

#-------------------------------------

def frank_bubbleSort(l):
    '''
    对上面的冒泡排序进行了优化
    :param l:
    :return:
    '''
    for i in range(len(l)):
        for j in range(len(l)-1, i+1, -1):
            if l[i] > l[j]:
                l[i], l[j] = l[j], l[i]

#-------------------------------------

def insertionSort(l):
    '''
    插入排序法，时间复杂度是O(n*n)
    :param l:
    :return:
    '''
    length = len(l)
    i = 1
    while i < length:
        tmp = l[i]
        j = i
        while (j > 0) and (tmp < l[j - 1]):
            l[j] = l[j - 1]
            j -= 1
        l[j] = tmp
        i += 1

#-------------------------------------

def shellSort(l):
    '''
    希尔排序
    :param l:
    :return:
    '''
    increment = len(l)
    while increment > 1:
        increment = increment // 3 + 1
        i = increment + 1
        while i <= len(l):
            if l[i] < l[i-increment]:
                l[0] = l[i]


#-------------------------------------

def quickSort(l):
    quickSortHelper(l, 0, len(l)-1)

def quickSortHelper(l, left, right):
    if left < right:
        pivotLocation = partition(l, left, right)
        quickSortHelper(l, left, pivotLocation-1)
        quickSortHelper(l, pivotLocation+1, right)

def partition(l, left, right):
    middle = (left + right) // 2
    pivot = l[middle]
    l[middle] = l[right]
    l[right] = pivot

    boundary = left
    for index in range(left, right):
        if l[index] < pivot:
            swap(l, index, boundary)
            boundary += 1

    swap(l, right, boundary)
    return boundary

def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = []
    middle = []
    right = []
    for i in arr:
        if i < pivot:
            left.append(i)
        elif i > pivot:
            right.append(i)
        else:
            middle.append(i)
    return quicksort(left) + middle + quicksort(right)

#-------------------------------------

def mergesort(seq):
    """归并排序"""
    if len(seq) <= 1:
        return seq
    mid = len(seq) / 2  # 将列表分成更小的两个列表
    # 分别对左右两个列表进行处理，分别返回两个排序好的列表
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    # 对排序好的两个列表合并，产生一个新的排序好的列表
    return merge(left, right)

def merge(left, right):
    """合并两个已排序好的列表，产生一个新的已排序好的列表"""
    result = []  # 新的已排序好的列表
    i = 0  # 下标
    j = 0
    # 对两个列表中的元素 两两对比。
    # 将最小的元素，放到result中，并对当前列表下标加1
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result

#-------------------------------------

def sift_down(arr, start, end):
    root = start
    while True:
        # 从root开始对最大堆调整
        child = 2 * root + 1
        if child > end:
            break
        # 找出两个child中交大的一个
        if child + 1 <= end and arr[child] < arr[child + 1]:
            child += 1

        if arr[root] < arr[child]:
            # 最大堆小于较大的child, 交换顺序
            arr[root], arr[child] = arr[child], arr[root]
            # 正在调整的节点设置为root
            root = child
        else:
            # 无需调整的时候, 退出
            break

def heap_sort(arr):
    # 从最后一个有子节点的孩子还是调整最大堆
    first = len(arr) // 2 - 1
    for start in range(first, -1, -1):
        sift_down(arr, start, len(arr) - 1)

    # 将最大的放到堆的最后一个, 堆-1, 继续调整排序
    for end in range(len(arr)-1, 0, -1):
        arr[0], arr[end] = arr[end], arr[0]
        sift_down(arr, 0, end - 1)

#-------------------------------------
def my_reverse(s):
    '''
    字符串反转
    :param s:
    :return:
    '''
    s = list(s)
    length = len(s)
    if length == 0:
        return
    start, end = 0, length-1
    while start < end:
        s[start], s[end] = s[end], s[start]
        start += 1
        end -= 1
    return ''.join(s)

#-------------------------------------
'''
求素数
'''
from itertools import count
def fun(n):
    return lambda x: x % n > 0
def primes():
    it = count(2) # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(fun(n), it) # 构造新序列

for i in primes():
    if i < 100:
        print(i)
    else:
        break
#-------------------------------------

