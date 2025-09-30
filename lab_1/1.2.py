from typing import List
from common.matrix import Matrix


def tridiagonal(A: Matrix, d: List[float]) -> List[float]:
    n = A.rows

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and A.matrix[i][j] != 0:
                raise ValueError(f"Матрица не является трехдиагональной. ")

    a = [0.0] * n
    b = [0.0] * n
    c = [0.0] * n

    for i in range(n):
        b[i] = A.matrix[i][i]
        if i > 0:
            a[i] = A.matrix[i][i - 1]
        if i < n - 1:
            c[i] = A.matrix[i][i + 1]

    P = [0.0] * n
    Q = [0.0] * n

    P[0] = -c[0] / b[0]
    Q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * P[i - 1]
        P[i] = -c[i] / denominator
        Q[i] = (d[i] - a[i] * Q[i - 1]) / denominator

    x = [0.0] * n
    x[n - 1] = Q[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = P[i] * x[i + 1] + Q[i]

    return x


def round_list(values: List[float], decimals: int = 10) -> List[float]:
    return [round(x, decimals) for x in values]


def main():
    A_data = [
        [-1, -1, 0, 0, 0],
        [7, -17, -8, 0, 0],
        [0, -9, 19, 8, 0],
        [0, 0, 7, -20, 4],
        [0, 0, 0, -4, 12]
    ]

    A = Matrix(A_data)
    d = [-4, 132, -59, -193, -40]

    print("Трехдиагональная матрица A:")
    print(A)
    print(f"\nВектор правой части d: {d}")

    try:
        solution = tridiagonal(A, d)
        rounded_solution = round_list(solution)
        print(f"\nРешение методом прогонки: {rounded_solution}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()