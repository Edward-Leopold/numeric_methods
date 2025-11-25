import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from typing import Callable, List, Tuple

# Добавляем путь к модулям из lab_1
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'lab_1'))

# Импортируем LUP-разложение
from lab_1.lup import lup_decomposition, solve_system_lup
from common.matrix import Matrix

# Параметр a = 3
a = 3

def f1(x1: float, x2: float) -> float:
    return x1 ** 2 / a ** 2 + x2 ** 2 / (a / 2) ** 2 - 1

def f2(x1: float, x2: float) -> float:
    return a * x2 - np.exp(x1) - x1

def phi1(x1: float, x2: float) -> float:
    term = 1 - x2 ** 2 / (a / 2) ** 2
    if term < 0:
        return x1  # Если корень отрицательный, оставляем текущее значение
    return np.sqrt(a ** 2 * term)

def phi2(x1: float, x2: float) -> float:
    return (np.exp(x1) + x1) / a

def jacobi(x1: float, x2: float) -> List[List[float]]:
    return [
        [2 * x1 / a ** 2, 2 * x2 / (a / 2) ** 2],
        [-np.exp(x1) - 1, a]
    ]

def eq_jacobi(x1: float, x2: float) -> List[List[float]]:
    """Якобиан для итерационных функций phi1 и phi2"""
    # d(phi1)/dx1 = 0 (phi1 не зависит от x1)
    dphi1_dx1 = 0

    # d(phi1)/dx2 = производная от sqrt(a²*(1 - x₂²/(a/2)²)) по x₂
    term = 1 - x2 ** 2 / (a / 2) ** 2
    if term <= 0:
        dphi1_dx2 = 0
    else:
        dphi1_dx2 = - (a ** 2 * x2 / ((a / 2) ** 2)) / (2 * np.sqrt(a ** 2 * term))

    # d(phi2)/dx1 = производная от (exp(x₁) + x₁)/a
    dphi2_dx1 = (np.exp(x1) + 1) / a

    # d(phi2)/dx2 = 0 (phi2 не зависит от x₂)
    dphi2_dx2 = 0

    return [
        [dphi1_dx1, dphi1_dx2],
        [dphi2_dx1, dphi2_dx2]
    ]

def system_newton_method(jacobi: Callable[..., List[List[float]]],
                         initial_approximation: List[float],
                         *equations: Callable[..., float],
                         accuracy: float = 1e-6) -> Tuple[List[float], int]:

    dimension = len(initial_approximation)
    x_prev = initial_approximation

    iterations = 0
    while True:
        # Создаем матрицу системы: J(x) * Δx = -F(x)
        J_data = jacobi(*x_prev)
        system = Matrix(J_data)

        # Вектор правой части: -F(x)
        free_members = [-eq(*x_prev) for eq in equations]

        # Решаем систему LUP * delta_x = free_members
        delta_x = solve_system_lup(system, free_members)

        # Вычисляем новое приближение
        x_next = [x_prev[i] + delta_x[i] for i in range(dimension)]

        # Проверяем критерии остановки
        delta_norm = max(abs(x_next[i] - x_prev[i]) for i in range(dimension))
        residual_norm = max(abs(eq(*x_next)) for eq in equations)

        if delta_norm <= accuracy and residual_norm <= accuracy:
            return x_next, iterations

        x_prev = x_next
        iterations += 1

def system_simple_iteration_method(eq_jacobi: Callable[..., List[List[float]]],
                                   initial_approximation: List[float],
                                   *eq_equations: Callable[..., float],
                                   accuracy: float = 1e-6) -> Tuple[List[float], int, List[float]]:
    """
    Метод простых итераций для систем уравнений
    """
    print(Matrix.calculate_norm(eq_jacobi(*initial_approximation)))
    q = 0
    eps = 0.2
    left_borders = []
    right_borders = []
    dimension = len(initial_approximation)

    for i in range(dimension):
        left_borders.append(initial_approximation[i] - eps)
        right_borders.append(initial_approximation[i] + eps)

    # Вычисляем q - максимальную норму якобиана в окрестности
    space = left_borders.copy()
    condition = True
    while condition:
        val = Matrix.calculate_norm(eq_jacobi(*space))
        print(val)
        if 1 > val > q:
            # print(q)
            q = val

        space = [space[i] + 1 / 10000 for i in range(dimension)]
        condition = all(right_borders[i] > space[i] for i in range(dimension))
    print(q)
    print(Matrix.calculate_norm(eq_jacobi(*initial_approximation)))
    # Основные итерации
    x_prev = initial_approximation
    iterations = 0

    while True:
        x_next = [eq(*x_prev) for eq in eq_equations]

        # Вычисляем погрешности для каждой компоненты
        fault = [(q / (1 - q)) * abs(x_next[i] - x_prev[i]) for i in range(dimension)]

        # Выводим информацию о погрешности (как в примере друга)
        print(f'fault = {[f"{f:.2e}" for f in fault]}')

        # Проверяем критерий остановки
        if all(f < accuracy for f in fault):
            return x_next, iterations, fault

        x_prev = x_next
        iterations += 1

def graph_1():
    """Построение графиков системы уравнений"""
    x1 = np.linspace(-4, 4, 400)
    x2 = np.linspace(-2, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z1 = f1(X1, X2)
    Z2 = f2(X1, X2)

    plt.figure(figsize=(8, 6))
    plt.contour(X1, X2, Z1, levels=[0], colors='blue')
    plt.contour(X1, X2, Z2, levels=[0], colors='red')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(f'Система уравнений (a={a})')
    plt.legend(['x1²/a² + x2²/(a/2)² - 1 = 0', f'a*x2 - exp(x1) - x1 = 0'])
    plt.grid(True)
    plt.show()

def main():
    # Построение графика
    graph_1()

    print("Введите начальное приближение x1 и x2:")
    print("Рекомендуется: 1.0 1.0")
    x1, x2 = map(float, input().split())
    eps = float(input("Введите погрешность: "))

    x0 = [x1, x2]

    print("\nМетод Ньютона:")
    try:
        solution, iterations = system_newton_method(jacobi, x0, f1, f2, accuracy=eps)
        print(f'solution = {[f"{x:.6f}" for x in solution]} iterations = {iterations}')
        print(f"Проверка: f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка: {e}")

    print("\nМетод простых итераций:")
    try:
        solution, iterations, fault = system_simple_iteration_method(eq_jacobi, x0, phi1, phi2, accuracy=eps)
        print(
            f'solution = {[f"{x:.6f}" for x in solution]} iterations = {iterations} fault = {[f"{f:.2e}" for f in fault]}')
        print(f"Проверка: f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка: {e}")

if __name__ == '__main__':
    main()