import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable
from common.matrix import Matrix
from lab_1.lup import lup_decomposition, solve_system_lup

class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    @staticmethod
    def calculate_norm(matrix_data: List[List[float]]) -> float:
        """Норма матрицы (максимальная сумма модулей элементов в строке)"""
        return max(sum(abs(x) for x in row) for row in matrix_data)


def solve_system_lup(matrix_obj, free_members):
    """Решение СЛАУ через numpy для совместимости"""
    A = np.array(matrix_obj.data, dtype=float)
    b = np.array(free_members, dtype=float)
    return np.linalg.solve(A, b).tolist()


# -----------------------------------------------------------------------------


# Параметр a = 3 (Вариант 14)
a = 3


def f1(x1: float, x2: float) -> float:
    # Уравнение 1: x1^2 / a^2 + x2^2 / (a/2)^2 - 1 = 0
    # При a=3: x1^2/9 + x2^2/2.25 - 1 = 0
    return (x1 ** 2) / (a ** 2) + (x2 ** 2) / ((a / 2) ** 2) - 1


def f2(x1: float, x2: float) -> float:
    # Уравнение 2: a*x2 - e^x1 - x1 = 0
    return a * x2 - np.exp(x1) - x1


def jacobi(x1: float, x2: float) -> List[List[float]]:
    """Матрица Якоби для метода Ньютона"""
    # df1/dx1 = 2*x1 / a^2
    df1_dx1 = 2 * x1 / (a ** 2)
    # df1/dx2 = 2*x2 / (a/2)^2 = 8*x2 / a^2
    df1_dx2 = 2 * x2 / ((a / 2) ** 2)

    # df2/dx1 = -e^x1 - 1
    df2_dx1 = -np.exp(x1) - 1
    # df2/dx2 = a
    df2_dx2 = a

    return [
        [df1_dx1, df1_dx2],
        [df2_dx1, df2_dx2]
    ]


# --- Функции для метода простых итераций ---
# Для сложной системы используем метод релаксации: x = x - lambda * f(x)
# Это эквивалентно x = phi(x), где phi' должна быть < 1.
LAMBDA = 0.1


def phi1(x1: float, x2: float) -> float:
    # x1 = x1 - lambda * f1(x1, x2)
    # Знак подбирается так, чтобы уменьшать невязку
    return x1 - LAMBDA * f1(x1, x2)


def phi2(x1: float, x2: float) -> float:
    # x2 = x2 + lambda * f2(x1, x2)
    # (знак зависит от знака производной, здесь пробуем + так как f2 = 3x2...)
    # f2 = 3x2 - ... Если f2 > 0, значит x2 слишком большой, надо уменьшать.
    # Значит x2 = x2 - lambda/a * f2
    return x2 - (LAMBDA / a) * f2(x1, x2)


def eq_jacobi(x1: float, x2: float) -> List[List[float]]:
    """Якобиан выражающих функций phi для оценки сходимости"""
    # d(phi1)/dx1 = 1 - lambda * df1/dx1
    dp1_dx1 = 1 - LAMBDA * (2 * x1 / (a ** 2))
    # d(phi1)/dx2 = - lambda * df1/dx2
    dp1_dx2 = - LAMBDA * (2 * x2 / ((a / 2) ** 2))

    # d(phi2)/dx1 = - (lambda/a) * df2/dx1
    dp2_dx1 = - (LAMBDA / a) * (-np.exp(x1) - 1)
    # d(phi2)/dx2 = 1 - (lambda/a) * df2/dx2 = 1 - (lambda/a)*a = 1 - lambda
    dp2_dx2 = 1 - LAMBDA

    return [[dp1_dx1, dp1_dx2], [dp2_dx1, dp2_dx2]]


def system_newton_method(jacobi_func: Callable, initial_approximation: List[float],
                         *equations: Callable, accuracy: float = 1e-6) -> Tuple[List[float], int]:
    dimension = len(initial_approximation)
    x_prev = initial_approximation
    iterations = 0

    # Ограничитель итераций на случай расходимости
    max_iter = 100

    while iterations < max_iter:
        J_data = jacobi_func(*x_prev)
        system_matrix = Matrix(J_data)
        # Внимание: для Ньютона решаем J * delta = -F
        free_members = [-eq(*x_prev) for eq in equations]

        try:
            delta_x = solve_system_lup(system_matrix, free_members)
        except np.linalg.LinAlgError:
            print("Матрица Якоби вырождена. Метод Ньютона не может продолжать.")
            return x_prev, iterations

        x_next = [x_prev[i] + delta_x[i] for i in range(dimension)]

        delta_norm = max(abs(x_next[i] - x_prev[i]) for i in range(dimension))

        # Проверка на сходимость
        if delta_norm <= accuracy:
            return x_next, iterations + 1

        x_prev = x_next
        iterations += 1

    print("Превышено максимальное число итераций в методе Ньютона.")
    return x_prev, iterations


def system_simple_iteration_method(eq_jacobi: Callable, initial_approximation: List[float],
                                   *eq_equations: Callable, accuracy: float = 1e-6) -> Tuple[
    List[float], int, List[float]]:
    dimension = len(initial_approximation)

    # Проверка условия сходимости в начальной точке
    # Если норма якобиана phi >= 1, метод не гарантирует сходимость
    norm = Matrix.calculate_norm(eq_jacobi(*initial_approximation))
    print(f"Norm of Phi Jacobian at start: {norm:.4f}")
    if norm >= 1:
        print("Внимание: Условие сходимости (norm < 1) не выполнено. Метод может разойтись.")

    # Используем q чуть меньше 1 для оценки ошибки, если норма плохая
    q = norm if norm < 1 else 0.99

    x_prev = initial_approximation
    iterations = 0
    max_iter = 500  # Защита от зацикливания

    while iterations < max_iter:
        x_next = [eq(*x_prev) for eq in eq_equations]

        # Оценка погрешности
        current_delta = max(abs(x_next[i] - x_prev[i]) for i in range(dimension))

        # Формула апостериорной оценки: ||x_k - x*|| <= q/(1-q) * ||x_k - x_{k-1}||
        # Если q близко к 0, оценка хорошая. Если q близко к 1, множитель велик.
        calc_error = (q / (1 - q)) * current_delta

        # Для вывода информации
        fault = [abs(x_next[i] - x_prev[i]) for i in range(dimension)]

        if current_delta < accuracy:
            return x_next, iterations + 1, fault

        x_prev = x_next
        iterations += 1

        # Если значения улетают в бесконечность
        if current_delta > 1e10:
            print("Метод простых итераций расходится.")
            break

    return x_prev, iterations, [0.0, 0.0]


def plot_system():
    # Диапазон построения графика
    x1 = np.linspace(-4, 4, 400)
    x2 = np.linspace(-3, 3, 400)
    X1, X2 = np.meshgrid(x1, x2)

    Z1 = f1(X1, X2)
    Z2 = f2(X1, X2)

    plt.figure(figsize=(10, 8))
    # Уравнение 1 (Эллипс)
    plt.contour(X1, X2, Z1, levels=[0], colors='blue', linewidths=2)
    # Уравнение 2 (Кривая)
    plt.contour(X1, X2, Z2, levels=[0], colors='red', linewidths=2)

    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(f'Система нелинейных уравнений (a={a})')
    # Легенда "костылем" для contour plot
    plt.plot([], [], color='blue', label=r'$\frac{x_1^2}{a^2} + \frac{x_2^2}{(a/2)^2} - 1 = 0$')
    plt.plot([], [], color='red', label=r'$ax_2 - e^{x_1} - x_1 = 0$')

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # Чтобы эллипс не выглядел сплюснутым
    plt.show()


def main():
    plot_system()

    print(f"Вариант 14 (a={a}). Решаем систему.")
    print("Судя по графику, положительное решение в районе x1=1.0, x2=1.3")
    print("Введите начальное приближение x1 и x2 (например: 1.0 1.3):")
    try:
        inp = input().split()
        if not inp:
            # Значения по умолчанию, если ничего не ввели
            x1, x2 = 1.0, 1.3
            print(f"Используются значения по умолчанию: {x1}, {x2}")
        else:
            x1, x2 = map(float, inp)

        eps_input = input("Введите погрешность (по умолчанию 1e-4): ")
        eps = float(eps_input) if eps_input else 1e-4
    except ValueError:
        print("Ошибка ввода. Используются стандартные значения.")
        x1, x2 = 1.0, 1.3
        eps = 1e-4

    print("\n--- Метод Ньютона ---")
    try:
        solution, iterations = system_newton_method(jacobi, [x1, x2], f1, f2, accuracy=eps)
        print(f'Решение: {[f"{x:.6f}" for x in solution]}')
        print(f'Итераций: {iterations}')
        print(f"Проверка (невязка): f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка в методе Ньютона: {e}")

    print("\n--- Метод простых итераций ---")
    try:
        # Для метода итераций используем phi1, phi2
        solution, iterations, fault = system_simple_iteration_method(eq_jacobi, [x1, x2], phi1, phi2, accuracy=eps)
        print(f'Решение: {[f"{x:.6f}" for x in solution]}')
        print(f'Итераций: {iterations}')
        print(f'Последняя разность: {[f"{f:.2e}" for f in fault]}')
        print(f"Проверка (невязка): f1 = {f1(solution[0], solution[1]):.2e}, f2 = {f2(solution[0], solution[1]):.2e}")
    except Exception as e:
        print(f"Ошибка в методе простых итераций: {e}")


if __name__ == '__main__':
    main()