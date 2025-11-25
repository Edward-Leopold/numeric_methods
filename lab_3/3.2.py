import math
from typing import List, Tuple
from lab_1.tridiagonal import tridiagonal
from common.matrix import Matrix


def make_system(X_values: List[float], F_values: List[float]) -> Tuple[List[float], Matrix, List[float]]:
    n = len(X_values)
    h = [X_values[i] - X_values[i - 1] for i in range(1, n)]

    A_data = [[0.0] * n for _ in range(n)]
    d = [0.0] * n

    # (левая граница)
    A_data[0][0] = 2
    A_data[0][1] = 1
    d[0] = 6 * ((F_values[1] - F_values[0]) / h[0]) / h[0]

    # Внутренние уравнения
    for i in range(1, n - 1):
        A_data[i][i - 1] = h[i - 1]
        A_data[i][i] = 2 * (h[i - 1] + h[i])
        A_data[i][i + 1] = h[i]
        d[i] = 6 * (
                (F_values[i + 1] - F_values[i]) / h[i] -
                (F_values[i] - F_values[i - 1]) / h[i - 1]
        )

    #  (правая граница)
    A_data[n - 1][n - 2] = 1
    A_data[n - 1][n - 1] = 2
    d[n - 1] = 6 * (-(F_values[n - 1] - F_values[n - 2]) / h[n - 2]) / h[n - 2]

    return h, Matrix(A_data), d


def spline_coefficients(X_values: List[float], F_values: List[float], h: List[float], M: List[float]) -> List[Tuple[float, float, float, float]]:
    n = len(F_values)
    coefficients = []

    for i in range(n - 1):
        a_i = F_values[i]
        b_i = (F_values[i + 1] - F_values[i]) / h[i] - h[i] * (2 * M[i] + M[i + 1]) / 6
        c_i = M[i] / 2
        d_i = (M[i + 1] - M[i]) / (6 * h[i])

        coefficients.append((a_i, b_i, c_i, d_i))

    return coefficients


def calc_spline(x: float, coefficients: List[Tuple[float, float, float, float]],
                X_values: List[float]) -> float:
    n = len(X_values)
    segment_index = -1

    for i in range(n - 1):
        if X_values[i] <= x <= X_values[i + 1]:
            segment_index = i
            break

    if segment_index == -1:
        if x < X_values[0]:
            segment_index = 0
        else:
            segment_index = n - 2

    a, b, c, d = coefficients[segment_index]
    left_border = X_values[segment_index]
    dx = x - left_border

    return a + b * dx + c * dx ** 2 + d * dx ** 3


def main():
    X = [0.0, 0.9, 1.8, 2.7, 3.6]
    F = [0.0, 0.72235, 1.5609, 2.8459, 7.7275]
    x_star = 1.5

    print("Кубический сплайн с нулевыми производными на концах")
    print("=" * 50)
    print("Узлы интерполяции:")
    for i, (x, f) in enumerate(zip(X, F)):
        print(f"x{i} = {x:.1f}, f{i} = {f:.4f}")

    print(f"\nТочка интерполяции: x* = {x_star}")

    h, A, d = make_system(X, F)
    M = tridiagonal(A, d)

    coefficients = spline_coefficients(X, F, h, M)
    result = calc_spline(x_star, coefficients, X)

    print(f"\nРезультат интерполяции: S({x_star}) = {result:.6f}")
    print("\nКоэффициенты сплайна для каждого отрезка:")
    for i, (a, b, c, d_coeff) in enumerate(coefficients):
        print(f"Отрезок [{X[i]:.1f}, {X[i + 1]:.1f}]:")
        print(f"  a{i} = {a:.4f}, b{i} = {b:.4f}, c{i} = {c:.4f}, d{i} = {d_coeff:.4f}")


if __name__ == '__main__':
    main()