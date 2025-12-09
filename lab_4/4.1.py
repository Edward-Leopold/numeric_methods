import math
from typing import Tuple, List, Callable
import numpy as np
import matplotlib.pyplot as plt


def exact_solution(x: int | float) -> int | float:
    """
    Точное решение, заданное в условии:
    y = (-0.9783 * cos(2x) + 0.4776 * sin(2x)) / sin(x)
    """
    numerator = -0.9783 * math.cos(2 * x) + 0.4776 * math.sin(2 * x)
    denominator = math.sin(x)

    return numerator / denominator


def f(x: int | float, y: int | float, z: int | float) -> int | float:
    """
    Исходное уравнение: y'' + 2y'ctgx + 3y = 0
    Приводим к системе:
    y' = z
    z' = -2*z*ctg(x) - 3*y

    Примечание: ctg(x) = cos(x)/sin(x).
    """
    # Используем cos/sin, чтобы избежать проблем с тангенсом в точке pi/2
    ctgx = math.cos(x) / math.sin(x)
    return -2 * z * ctgx - 3 * y


def euler_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """Явный метод Эйлера."""
    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]

    n_steps = int(round((b - a) / h))

    x_current = x0
    y_current = y0
    z_current = z0

    for _ in range(n_steps):
        # Защита от выхода за границу из-за погрешности float
        if x_current + h > b + 1e-9:
            break

        y_next = y_current + h * z_current
        z_next = z_current + h * func(x_current, y_current, z_current)
        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values

def euler_cauchy_method(space: Tuple[int | float, int | float], h: float | int,
                        initial_condition: Tuple[int | float, int | float, int | float, int | float],
                        func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """Метод Эйлера-Коши."""
    x0, y0, _, z0 = initial_condition
    a, b = space
    x_values = [x0]
    y_values = [y0]
    n_steps = int(round((b - a) / h))
    x_current = x0
    y_current = y0
    z_current = z0
    for _ in range(n_steps):
        if x_current + h > b + 1e-9:
            break
        y_tilde = y_current + h * z_current
        z_tilde = z_current + h * func(x_current, y_current, z_current)
        y_next = y_current + h * 0.5 * (z_current + z_tilde)
        z_next = z_current + h * 0.5 * (
            func(x_current, y_current, z_current) +
            func(x_current + h, y_tilde, z_tilde)
        )
        x_next = x_current + h
        x_values.append(x_next)
        y_values.append(y_next)
        x_current, y_current, z_current = x_next, y_next, z_next
    return x_values, y_values

def improved_euler_method(space: Tuple[int | float, int | float], h: float | int,
                          initial_condition: Tuple[int | float, int | float, int | float, int | float],
                          func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """Улучшенный метод Эйлера."""
    x0, y0, _, z0 = initial_condition
    a, b = space
    x_values = [x0]
    y_values = [y0]
    n_steps = int(round((b - a) / h))
    x_current = x0
    y_current = y0
    z_current = z0
    for _ in range(n_steps):
        if x_current + h > b + 1e-9:
            break
        y_half = y_current + (h / 2) * z_current
        z_half = z_current + (h / 2) * func(x_current, y_current, z_current)
        x_half = x_current + h / 2
        y_next = y_current + h * z_half
        z_next = z_current + h * func(x_half, y_half, z_half)
        x_next = x_current + h
        x_values.append(x_next)
        y_values.append(y_next)
        x_current, y_current, z_current = x_next, y_next, z_next
    return x_values, y_values

def runge_kutta_method(space: Tuple[int | float, int | float], h: float | int,
                       initial_condition: Tuple[int | float, int | float, int | float, int | float],
                       func: Callable[[int | float, int | float, int | float], int | float], p: int) \
        -> Tuple[List[int | float], List[int | float], List[int | float]]:
    """Метод Рунге-Кутты (поддерживает 3 и 4 порядок)."""
    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    x_current, y_current, z_current = x0, y0, z0
    n_steps = int(round((b - a) / h))

    for _ in range(n_steps):
        if x_current + h > b + 1e-9:
            break

        K1_y = h * z_current
        K1_z = h * func(x_current, y_current, z_current)

        if p == 4:
            K2_y = h * (z_current + K1_z / 2)
            K2_z = h * func(x_current + h / 2,
                            y_current + K1_y / 2,
                            z_current + K1_z / 2)

            K3_y = h * (z_current + K2_z / 2)
            K3_z = h * func(x_current + h / 2,
                            y_current + K2_y / 2,
                            z_current + K2_z / 2)

            K4_y = h * (z_current + K3_z)
            K4_z = h * func(x_current + h,
                            y_current + K3_y,
                            z_current + K3_z)

            y_next = y_current + (K1_y + 2 * K2_y + 2 * K3_y + K4_y) / 6
            z_next = z_current + (K1_z + 2 * K2_z + 2 * K3_z + K4_z) / 6

        else:
            raise ValueError("Supported orders: 4")

        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)
        z_values.append(z_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values, z_values


def adams_method(space: Tuple[int | float, int | float], h: float | int,
                                   initial_condition: Tuple[int | float, int | float, int | float, int | float],
                                   func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """Метод Адамса 4-го порядка (предиктор-корректор)."""
    x0, y0, _, z0 = initial_condition
    a, b = space

    # Разгон методом Рунге-Кутты 4-го порядка для первых 4 точек
    x_rk, y_rk, z_rk = runge_kutta_method(space, h, initial_condition, func, 4)

    X = x_rk[:4]
    Y = y_rk[:4]
    Z = z_rk[:4]
    # Вычисляем производные z' для первых точек
    F = [func(x, y, z) for x, y, z in zip(X, Y, Z)]

    n_steps = int(round((b - a) / h))

    for k in range(3, n_steps):
        if X[-1] + h > b + 1e-9:
            break

        x_new = X[k] + h

        # Предиктор (Адамс-Бэшфорт)
        y_new = Y[k] + h / 24 * (55 * Z[k] - 59 * Z[k - 1] + 37 * Z[k - 2] - 9 * Z[k - 3])
        z_new = Z[k] + h / 24 * (55 * F[k] - 59 * F[k - 1] + 37 * F[k - 2] - 9 * F[k - 3])

        X.append(x_new)
        Y.append(y_new)
        Z.append(z_new)
        F.append(func(x_new, y_new, z_new))

    return X, Y


def runge_romberg_error_estimation(space: Tuple[float, float], h: float,
                                   solutions: Tuple[List[float], List[float], List[float], List[float]], p: int = 1
                                   ) -> List[int | float]:
    """Оценка погрешности методом Рунге-Ромберга."""
    x_h, y_h, x_2h, y_2h = solutions

    results = []

    for i, x_val in enumerate(x_2h):
        # Ищем соответствующий индекс в массиве с шагом h
        idx_h = -1
        for idx, val in enumerate(x_h):
            if abs(val - x_val) < 1e-8:
                idx_h = idx
                break

        if idx_h != -1:
            y_h_val = y_h[idx_h]
            y_2h_val = y_2h[i]

            # Формула Рунге-Ромберга
            R_h = (y_h_val - y_2h_val) / (2 ** p - 1)
            results.append((x_val, y_h_val, R_h))

    return results


def print_results(X: List[int | float], Y: List[int | float],
                  runge_estimates: List[int | float], method_name: str) -> None:
    """Вывод таблиц результатов."""
    runge_dict = {x: (y_h, R_h) for x, y_h, R_h in runge_estimates}

    print(f"\n{method_name:^80}")
    print("=" * 80)
    print(f"{'x':<8} {'y_approx':<12} {'y_exact':<12} {'Abs Error':<15} "
          f"{'Rel Error':<15} {'R_h':<15}")
    print("=" * 80)

    for i, (x, y_approx) in enumerate(zip(X, Y)):
        y_exact = exact_solution(x)
        abs_error = abs(y_approx - y_exact)
        rel_error = abs_error / abs(y_exact) if abs(y_exact) > 1e-12 else 0

        r_val = "N/A"

        # Ищем R_h для текущего X (с учетом погрешности float)
        found_rh = False
        for rx in runge_dict:
            if abs(rx - x) < 1e-8:
                _, r_num = runge_dict[rx]
                r_val = f"{r_num:.6e}"
                found_rh = True
                break

        print(f"{x:<8.2f} {y_approx:<12.6f} {y_exact:<12.6f} "
              f"{abs_error:<15.6e} {rel_error:<15.6e} {r_val:<15}")

    print()


def main():
    initial_condition = (1, 1, 0, 1)
    space = (1, 2)
    h = 0.1

    results = {}

    # Метод Эйлера
    X_h, Y_h = euler_method(space, h, initial_condition, f)
    X_2h, Y_2h = euler_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, p=1)
    print_results(X_h, Y_h, errors, "Метод Эйлера")
    results['Euler'] = (X_h, Y_h)

    # Метод Эйлера-Коши
    X_h, Y_h = euler_cauchy_method(space, h, initial_condition, f)
    X_2h, Y_2h = euler_cauchy_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 2)
    print_results(X_h, Y_h, errors, "Метод Эйлера-Коши")
    results['Euler-Cauchy'] = (X_h, Y_h)

    # Улучшенный метод Эйлера
    X_h, Y_h = improved_euler_method(space, h, initial_condition, f)
    X_2h, Y_2h = improved_euler_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 2)
    print_results(X_h, Y_h, errors, "Улучшенный метод Эйлера")
    results['Improved Euler'] = (X_h, Y_h)

    # Метод Рунге-Кутты (4 порядка)
    X_h, Y_h, _ = runge_kutta_method(space, h, initial_condition, f, 4)
    X_2h, Y_2h, _ = runge_kutta_method(space, 2 * h, initial_condition, f, 4)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, p=4)
    print_results(X_h, Y_h, errors, "Метод Рунге-Кутты 4 порядка")
    results['Runge-Kutta 4'] = (X_h, Y_h)

    # Метод Адамса (4 порядка)
    X_h, Y_h = adams_method(space, h, initial_condition, f)
    X_2h, Y_2h = adams_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, p=4)
    print_results(X_h, Y_h, errors, "Метод Адамса 4-го порядка")
    results['Adams 4'] = (X_h, Y_h)

    # Plotting solutions
    X_fine = np.linspace(1, 2, 100)
    Y_exact = [exact_solution(x) for x in X_fine]

    plt.figure(figsize=(10, 6))
    plt.plot(X_fine, Y_exact, label='Exact Solution', linewidth=2)

    for method, (X, Y) in results.items():
        plt.plot(X, Y, label=method, marker='o')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparison of Numerical Methods with Exact Solution')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()