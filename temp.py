from math import sqrt
def quadratic(a, b, c):
    '''
    求二元一次方程的解

    '''
    if not isinstance(a, (int, float)):
        raise TypeError('type error')
    if not isinstance(b, (int, float)):
        raise TypeError('type error')
    if not isinstance(c, (int, float)):
        raise TypeError('type error')
    if a == 0:
        print('该方程不是二次方程!!!')
        return
    delta = b**2 - 4*a*c
    if delta < 0:
        return
    elif delta == 0:
        return -b / (2*a)
    else:
        x1 = (-b + sqrt(delta)) / (2*a)
        x2 = (-b - sqrt(delta)) / (2*a)
        return x1, x2


def power(x, n=2):
    '''
    计算一个数的次方
    '''
    if not isinstance(x, (int, float)):
        raise TypeError('type error')
    if not isinstance(n, int):
        raise TypeError('type error')
    if x == 0:
        return 0
    if n == 0:
        return 1
    result = 1
    m = abs(n)

    while m:
        result *= x
        m -= 1
    return result if n > 0 else 1 / result

def my_power(x, n):
    a = abs(n)
    res = 1
    while a != 0:
        if a % 2 != 0:
            res *= x
        x *= x
        a = a // 2
    return res if n >= 0 else 1 / res



def trim(s):
    '''
    去除字符串首尾的空格
    '''
    if not isinstance(s, str):
        raise TypeError('type error')
    start = 0
    end = len(s) - 1
    for i in range(len(s)):
        if s[i] != ' ':
            start = i
            break
    for j in range(len(s) - 1, -1, -1):
        if s[j] != ' ':
            end = j
            break
    if start == 0 and end == len(s)-1:
        return ''
    else:
        return s[start:end+1]


def findMinAndMax(l):
    '''
    查找list中的最大值和最小值
    '''
    min_num = 0xffffff
    max_num = -0xffffff
    for i in l:
        if i < min_num:
            min_num = i
        if i > max_num:
            max_num = i
    return (min_num, max_num)


def triangles():
    '''
    生成器：杨辉三角
    '''
    L = [1]
    while True:
        yield L
        L = [1] + [L[i] +L[i+1] for i in range(len(L) - 1)] + [1]


# 所有的for循环背后都是这么做的
it = iter([1,2,3,4,5])
while True:
    try:
        x = next(it)
        print(x)
    except StopIteration as e:
        break

#---------------------------------------------------------
from functools import reduce
digits = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
def char2int(c):
    try:
        return digits[c]
    except:
        raise TypeError('type error')
def str2int(s):
    # 例子：把'1234'转化成1234
    return reduce(lambda x, y: 10*x+y, map(char2num, s))
def str2float(s):
    a, b = s.split('.')
    a = str2int(a)
    length = len(b)
    l = [0] + list(map(char2int, b))
    l.reverse()
    res = reduce(lambda x, y: 0.1*x+y, l)
    return a + round(res, length)
#---------------------------------------------------------
def nature_number():
    n = 2
    while True:
        yield n
        n += 1
def fun(n):
    return lambda x: x % n > 0
def primes():
    it = nature_number() # 初始序列
    while True:
        n = next(it) # 返回序列的第一个数
        yield n
        it = filter(fun(n), it) # 构造新序列

for i in primes():
    if i < 100:
        print(i)
    else:
        break
#---------------------------------------------------------








