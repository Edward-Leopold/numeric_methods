import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from common.matrix import Matrix
from lab_1.lup import lup_decomposition, solve_system_lup

# Параметр a = 2
a = 2


def f1(x1: float, x2: float) -> float:
    return a * x1 ** 2 - x1 + x2 ** 2 - 1


def f2(x1: float, x2: float) -> float:
    return x2 - np.tan(x1)


def jacobi(x1: float, x2: float) -> List[List[float]]:
    return [
        [2 * a * x1 - 1, 2 * x2],
        [-1 / np.cos(x1) ** 2, 1]
    ]


def phi1(x1: float, x2: float) -> float:
    return np.sqrt((1 - x2 ** 2 + x1) / a) if (1 - x2 ** 2 + x1) >= 0 else x1


def phi2(x1: float, x2: float) -> float:
    return np.tan(x1)


def eq_jacobi(x1: float, x2: float) -> List[List[float]]:
    dphi1_dx1 = 1 / (2 * a * np.sqrt((1 - x2 ** 2 + x1) / a)) if (1 - x2 ** 2 + x1) > 0 else 1
    dphi1_dx2 = -x2 / (a * np.sqrt((1 - x2 ** 2 + x1) / a)) if (1 - x2 ** 2 + x1) > 0 else 0
    dphi2_dx1 = 1 / np.cos(x1) ** 2
    dphi2_dx2 = 0
    return [[dphi1_dx1, dphi1_dx2], [dphi2_dx1, dphi2_dx2]]


def system_newton_method(jacobi_func: Callable, initial_approximation: List[float],
                         *equations: Callable, accuracy: float = 1e-6) -> Tuple[List[float], int]:
    dimension = len(initial_approximation)
    x_prev = initial_approximation
    iterations = 0

    while True:
        J_data = jacobi_func(*x_prev)
        system_matrix = Matrix(J_data)
        free_members = [-eq(*x_prev) for eq in equations]

        delta_x = solve_system_lup(system_matrix, free_members)

        x_next = [x_prev[i] + delta_x[i] for i in range(dimension)]

        delta_norm = max(abs(x_next[i] - x_prev[i]) for i in range(dimension))
        residual_norm = max(abs(eq(*x_next)) for eq in equations)

        if delta_norm <= accuracy and residual_norm <= accuracy:
            return x_next, iterations

        x_prev = x_next
        iterations += 1


def system_simple_iteration_method(eq_jacobi: Callable, initial_approximation: List[float],
                                   *eq_equations: Callable, accuracy: float = 1e-6) -> Tuple[
    List[float], int, List[float]]:
    q = 0
    eps = 0.2
    dimension = len(initial_approximation)

    left_borders = [initial_approximation[i] - eps for i in range(dimension)]
    right_borders = [initial_approximation[i] + eps for i in range(dimension)]

    space = left_borders.copy()
    condition = True
    while condition:
        val = Matrix.calculate_norm(eq_jacobi(*space))
        if 1 > val > q:
            q = val
        space = [space[i] + 1 / 10000 for i in range(dimension)]
        condition = all(right_borders[i] > space[i] for i in range(dimension))

    x_prev = initial_approximation
    iterations = 0

    while True:
        x_next = [eq(*x_prev) for eq in eq_equations]
        fault = [(q / (1 - q)) * abs(x_next[i] - x_prev[i]) for i in range(dimension)]
        print(f'fault = {[f"{f:.2e}" for f in fault]}')

        if all(f < accuracy for f in fault):
            return x_next, iterations, fault

        x_prev = x_next
        iterations += 1


def plot_system():
    x1 = np.linspace(-2, 2, 400)
    x2 = np.linspace(-2, 2, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z1 = f1(X1, X2)
    Z2 = f2(X1, X2)

    plt.figure(figsize=(10, 8))
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2)
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2)
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'Система уравнений (a={a})')
    plt.legend([f'{a}x₁² - x₁ + x₂² - 1 = 0', 'x₂ - tg(x₁) = 0'])
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


def main():
    plot_system()

    print("Введите начальное приближение x1 и x2:")
    x1, x2 = map(float, input().split())
    eps = float(input("Введите погрешность: "))

    print("\nМетод Ньютона:")
    try:
        solution, iterations = system_newton_method(jacobi, [x1, x2], f1, f2, accuracy=eps)
        print(f'solution = {[f"{x:.6f}" for x in solution]} iterations = {iterations}')
        print(f"Проверка: f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\nМетод простых итераций:")
    try:
        solution, iterations, fault = system_simple_iteration_method(eq_jacobi, [x1, x2], phi1, phi2, accuracy=eps)
        print(
            f'solution = {[f"{x:.6f}" for x in solution]} iterations = {iterations} fault = {[f"{f:.2e}" for f in fault]}')
        print(f"Проверка: f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()