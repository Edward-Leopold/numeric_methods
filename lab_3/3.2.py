import math
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np


def tridiagonal(A, d):
    n = len(d)
    a = [0.0] + [A.data[i][i - 1] for i in range(1, n)]
    b = [A.data[i][i] for i in range(n)]
    c = [A.data[i][i + 1] for i in range(n - 1)] + [0.0]

    alpha = [0.0] * n
    beta = [0.0] * n

    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n - 1):
        alpha[i] = -c[i] / (b[i] + a[i] * alpha[i - 1])
        beta[i] = (d[i] - a[i] * beta[i - 1]) / (b[i] + a[i] * alpha[i - 1])

    beta[n - 1] = (d[n - 1] - a[n - 1] * beta[n - 2]) / (b[n - 1] + a[n - 1] * alpha[n - 2])

    x = [0.0] * n
    x[n - 1] = beta[n - 1]

    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


class Matrix:
    def __init__(self, data):
        self.data = data


def make_system(X_values: List[float], F_values: List[float]) -> Tuple[List[float], Matrix, List[float]]:
    n = len(X_values)
    h = [X_values[i] - X_values[i - 1] for i in range(1, n)]

    A_data = [[0.0] * n for _ in range(n)]
    d = [0.0] * n
    A_data[0][0] = 2
    A_data[0][1] = 1
    d[0] = 6 * ((F_values[1] - F_values[0]) / h[0]) / h[0]

    for i in range(1, n - 1):
        A_data[i][i - 1] = h[i - 1]
        A_data[i][i] = 2 * (h[i - 1] + h[i])
        A_data[i][i + 1] = h[i]
        d[i] = 6 * (
                (F_values[i + 1] - F_values[i]) / h[i] -
                (F_values[i] - F_values[i - 1]) / h[i - 1]
        )

    A_data[n - 1][n - 2] = 1
    A_data[n - 1][n - 1] = 2
    d[n - 1] = 6 * (-(F_values[n - 1] - F_values[n - 2]) / h[n - 2]) / h[n - 2]

    return h, Matrix(A_data), d


def spline_coefficients(X_values: List[float], F_values: List[float], h: List[float], M: List[float]) -> List[
    Tuple[float, float, float, float]]:
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


def plot_spline(X: List[float], F: List[float], coefficients: List[Tuple[float, float, float, float]], x_star: float):
    plt.figure(figsize=(10, 6))

    x_min = min(X)
    x_max = max(X)
    x_plot = np.linspace(x_min, x_max, 400)
    y_plot = [calc_spline(xi, coefficients, X) for xi in x_plot]

    plt.plot(x_plot, y_plot, 'b-', label='Кубический сплайн', linewidth=2)
    plt.scatter(X, F, color='red', s=100, zorder=5, label='Узлы интерполяции')

    y_star = calc_spline(x_star, coefficients, X)
    plt.scatter([x_star], [y_star], color='green', s=150, zorder=6,
                label=f'x* = {x_star}', marker='*')

    for i in range(len(X) - 1):
        x_seg = np.linspace(X[i], X[i + 1], 20)
        y_seg = [calc_spline(xi, coefficients, X) for xi in x_seg]
        plt.plot(x_seg, y_seg, linewidth=3, alpha=0.3)

    plt.xlabel('x')
    plt.ylabel('S(x)')
    plt.title('Кубический сплайн с нулевыми производными на концах')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


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

    plot_spline(X, F, coefficients, x_star)


if __name__ == '__main__':
    main()