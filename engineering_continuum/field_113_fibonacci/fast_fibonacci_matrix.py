"""Course 113: Fast Matrix Exponentiation O(log N) Fibonacci & Lucas Generator"""
class FastFibonacciMatrix:
    @staticmethod
    def _matrix_mult(a: list, b: list) -> list:
        return [
            [a[0][0]*b[0][0] + a[0][1]*b[1][0], a[0][0]*b[0][1] + a[0][1]*b[1][1]],
            [a[1][0]*b[0][0] + a[1][1]*b[1][0], a[1][0]*b[0][1] + a[1][1]*b[1][1]]
        ]

    @classmethod
    def _matrix_power(cls, mat: list, n: int) -> list:
        res = [[1, 0], [0, 1]]
        base = mat
        while n > 0:
            if n % 2 == 1:
                res = cls._matrix_mult(res, base)
            base = cls._matrix_mult(base, base)
            n //= 2
        return res

    @classmethod
    def fibonacci(cls, n: int) -> int:
        if n <= 0:
            return 0
        mat = [[1, 1], [1, 0]]
        res = cls._matrix_power(mat, n - 1)
        return res[0][0]
