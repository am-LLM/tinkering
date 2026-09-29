from fast_fibonacci_matrix import FastFibonacciMatrix

def test_fast_fibonacci():
    assert FastFibonacciMatrix.fibonacci(0) == 0
    assert FastFibonacciMatrix.fibonacci(1) == 1
    assert FastFibonacciMatrix.fibonacci(10) == 55
    assert FastFibonacciMatrix.fibonacci(20) == 6765
