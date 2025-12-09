from typing import List, Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np

from common.matrix import Matrix
from lab_1.lup import lup_decomposition, solve_system_lup


def build_system(X_values: List[float], Y_values: List[float], degree: int) -> Tuple[Matrix, List[float]]:
    """
    Построение системы для МНК
    """
    n = len(X_values)
    matrix = []
    free_terms = []

    for i in range(degree):
        row = []
        for j in range(degree):
            power = i + j
            row.append(sum(x ** power for x in X_values))
        matrix.append(row)

        free_term = sum(Y_values[k] * (X_values[k] ** i) for k in range(n))
        free_terms.append(free_term)

    return Matrix(matrix), free_terms


def get_polynomial(X_values: List[float], Y_values: List[float], degree: int) -> List[float]:
    """
    Получение коэффициентов полинома методом МНК
    """
    system, free_terms = build_system(X_values, Y_values, degree)
    return solve_system_lup(system, free_terms)


def compute_polynomial(coeffs: List[float], x: float) -> float:
    """
    Вычисление значения полинома в точке x
    """
    return sum(coef * (x ** i) for i, coef in enumerate(coeffs))


def compute_error(coeffs: List[float], X_values: List[float], Y_values: List[float]) -> float:
    """
    Вычисление суммы квадратов ошибок
    """
    total = 0.0
    for i in range(len(X_values)):
        predicted = compute_polynomial(coeffs, X_values[i])
        total += (predicted - Y_values[i]) ** 2
    return total


def polynomial_string(coeffs: List[float]) -> str:
    """
    Формирование строки полинома
    """
    terms = []
    for i, coef in enumerate(coeffs):
        if i == 0:
            terms.append(f"{coef:.4f}")
        elif i == 1:
            terms.append(f"{coef:.4f}x")
        else:
            terms.append(f"{coef:.4f}x^{i}")
    return " + ".join(terms)


def main():
    # Данные из задания
    X = [-0.9, 0.0, 0.9, 1.8, 2.7, 3.6]
    Y = [-1.2689, 0.0, 1.2689, 2.6541, 4.4856, 9.9138]

    # Полином 1-й степени
    coeffs_linear = get_polynomial(X, Y, 2)
    error_linear = compute_error(coeffs_linear, X, Y)

    # Полином 2-й степени
    coeffs_quadratic = get_polynomial(X, Y, 3)
    error_quadratic = compute_error(coeffs_quadratic, X, Y)

    print("Аппроксимация методом наименьших квадратов")
    print("=" * 50)
    print(f"Полином 1-й степени: y = {polynomial_string(coeffs_linear)}")
    print(f"Сумма квадратов ошибок: {error_linear:.6f}")
    print()
    print(f"Полином 2-й степени: y = {polynomial_string(coeffs_quadratic)}")
    print(f"Сумма квадратов ошибок: {error_quadratic:.6f}")

    # Построение графиков
    x_plot = np.linspace(min(X) - 0.5, max(X) + 0.5, 200)

    y_linear = [compute_polynomial(coeffs_linear, x) for x in x_plot]
    y_quadratic = [compute_polynomial(coeffs_quadratic, x) for x in x_plot]

    plt.figure(figsize=(10, 6))

    # Исходные точки
    plt.scatter(X, Y, color='black', s=60, zorder=5, label='Исходные данные')

    # Графики полиномов
    plt.plot(x_plot, y_linear, 'blue', linewidth=2, label='Полином 1-й степени')
    plt.plot(x_plot, y_quadratic, 'red', linewidth=2, label='Полином 2-й степени')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация методом наименьших квадратов')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()