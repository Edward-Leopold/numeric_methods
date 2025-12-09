import math
from typing import Tuple, List, Callable


# ---------------------------------------------------------
# Вспомогательные функции и точное решение
# ---------------------------------------------------------

def exact_solution(x: float) -> float:
    """
    Точное решение: y = e^x - 1
    """
    return math.exp(x) - 1


def p_func(x: float) -> float:
    return math.exp(x) + 1


def q_func(x: float) -> float:
    return -2.0


def r_func(x: float) -> float:
    return -math.exp(x)


def f_func(x: float) -> float:
    return 0.0


def ode_system(x: float, y: float, z: float) -> float:
    """
    Представление уравнения y'' = ... в виде системы для Рунге-Кутты.
    (e^x + 1)y'' - 2y' - e^x y = 0
    => y'' = (2y' + e^x y) / (e^x + 1)

    z = y'
    z' = (2*z + e^x * y) / (e^x + 1)
    """
    numerator = 2 * z + math.exp(x) * y
    denominator = math.exp(x) + 1
    return numerator / denominator


# ---------------------------------------------------------
# Метод Рунге-Кутты (для метода стрельбы)
# ---------------------------------------------------------

def runge_kutta_solve(space: Tuple[float, float], h: float, y0: float, z0: float) \
        -> Tuple[List[float], List[float], List[float]]:
    """
    Решает задачу Коши методом РК-4.
    Возвращает массивы X, Y, Z (где Z = y').
    """
    a, b = space
    steps = int(round((b - a) / h))

    X = [a]
    Y = [y0]
    Z = [z0]

    x_curr, y_curr, z_curr = a, y0, z0

    for _ in range(steps):
        k1_y = h * z_curr
        k1_z = h * ode_system(x_curr, y_curr, z_curr)

        k2_y = h * (z_curr + 0.5 * k1_z)
        k2_z = h * ode_system(x_curr + 0.5 * h, y_curr + 0.5 * k1_y, z_curr + 0.5 * k1_z)

        k3_y = h * (z_curr + 0.5 * k2_z)
        k3_z = h * ode_system(x_curr + 0.5 * h, y_curr + 0.5 * k2_y, z_curr + 0.5 * k2_z)

        k4_y = h * (z_curr + k3_z)
        k4_z = h * ode_system(x_curr + h, y_curr + k3_y, z_curr + k3_z)

        y_next = y_curr + (k1_y + 2 * k2_y + 2 * k3_y + k4_y) / 6
        z_next = z_curr + (k1_z + 2 * k2_z + 2 * k3_z + k4_z) / 6
        x_next = x_curr + h

        X.append(x_next)
        Y.append(y_next)
        Z.append(z_next)

        x_curr, y_curr, z_curr = x_next, y_next, z_next

    return X, Y, Z


# ---------------------------------------------------------
# 1. Метод Стрельбы
# ---------------------------------------------------------

def shooting_method(space: Tuple[float, float], h: float) -> Tuple[List[float], List[float]]:
    """
    Решение краевой задачи методом стрельбы.
    Условия:
    Left: y'(0) = 1
    Right: y'(1) - y(1) = 1

    В задаче Коши неизвестно y(0). Обозначим y(0) = eta.
    Тогда y'(0) = 1 (из условия).
    Мы ищем eta, чтобы выполнялось F(eta) = y'(1, eta) - y(1, eta) - 1 = 0.
    """
    a, b = space

    # Функция невязки (ошибки) на правой границе
    def residual(eta):
        # Решаем задачу Коши с y(0) = eta, y'(0) = 1
        _, Y, Z = runge_kutta_solve(space, h, y0=eta, z0=1.0)
        y_end = Y[-1]
        z_end = Z[-1]  # это y'(1)
        # Условие: y'(1) - y(1) = 1 => y'(1) - y(1) - 1 = 0
        return z_end - y_end - 1.0

    # Метод секущих для поиска корня уравнения residual(eta) = 0
    eta0 = 0.5  # Первое приближение
    eta1 = 1.5  # Второе приближение

    eps = 1e-6
    max_iter = 100

    eta_res = eta1

    for _ in range(max_iter):
        r0 = residual(eta0)
        r1 = residual(eta1)

        if abs(r1 - r0) < 1e-12:
            break

        # Формула секущих
        eta_next = eta1 - r1 * (eta1 - eta0) / (r1 - r0)

        if abs(eta_next - eta1) < eps:
            eta_res = eta_next
            break

        eta0, eta1 = eta1, eta_next
        eta_res = eta_next

    # Получаем финальное решение с найденным eta
    return runge_kutta_solve(space, h, y0=eta_res, z0=1.0)[:2]


# ---------------------------------------------------------
# 2. Конечно-разностный метод
# ---------------------------------------------------------

def finite_difference_method(space: Tuple[float, float], h: float) -> Tuple[List[float], List[float]]:
    """
    Решение краевой задачи методом конечных разностей.
    Используется аппроксимация O(h^2).
    """
    a, b = space
    N = int(round((b - a) / h))
    X = [a + i * h for i in range(N + 1)]

    # Коэффициенты для метода прогонки: A[i]*y[i-1] + B[i]*y[i] + C[i]*y[i+1] = D[i]
    # Индексация массивов прогонки: 0..N
    Alpha = [0.0] * (N + 1)  # A
    Beta = [0.0] * (N + 1)  # B
    Gamma = [0.0] * (N + 1)  # C
    Phi = [0.0] * (N + 1)  # D (правая часть)

    # Заполняем внутренние узлы (от 1 до N-1)
    # Уравнение: P*y'' + Q*y' + R*y = F
    # Аппроксимация:
    # y'' ~ (y_{i+1} - 2y_i + y_{i-1}) / h^2
    # y'  ~ (y_{i+1} - y_{i-1}) / 2h
    # Приводим к виду: y_{i-1}(P - Qh/2) + y_i(-2P + Rh^2) + y_{i+1}(P + Qh/2) = Fh^2

    for i in range(1, N):
        x = X[i]
        P = p_func(x)
        Q = q_func(x)
        R = r_func(x)
        F = f_func(x)

        Alpha[i] = P - Q * h / 2.0
        Beta[i] = -2 * P + R * h ** 2
        Gamma[i] = P + Q * h / 2.0
        Phi[i] = F * h ** 2

    # --- Граничные условия (Аппроксимация O(h^2)) ---

    # Левая граница x=0: y'(0) = 1
    # Используем фиктивный узел y_{-1}: (y_1 - y_{-1}) / 2h = 1 => y_{-1} = y_1 - 2h
    # Подставляем в уравнение для i=0:
    # A_0(y_1 - 2h) + B_0 y_0 + C_0 y_1 = 0 (т.к F=0)
    # y_0 * B_0 + y_1 * (A_0 + C_0) = 2h * A_0

    P0 = p_func(X[0])
    Q0 = q_func(X[0])
    R0 = r_func(X[0])

    A0 = P0 - Q0 * h / 2.0
    B0 = -2 * P0 + R0 * h ** 2
    C0 = P0 + Q0 * h / 2.0

    # Коэффициенты прогонки для 0-й строки (y_0*b + y_1*c = d)
    # В стандартной нотации прогонки у нас a_i y_{i-1}, здесь a_0 нет.
    Beta[0] = B0
    Gamma[0] = A0 + C0
    Phi[0] = 2 * h * A0

    # Правая граница x=1: y'(1) - y(1) = 1
    # Фиктивный узел y_{N+1}: (y_{N+1} - y_{N-1}) / 2h - y_N = 1
    # y_{N+1} = y_{N-1} + 2h(1 + y_N)
    # Подставляем в уравнение для i=N:
    # A_N y_{N-1} + B_N y_N + C_N (y_{N-1} + 2h(1 + y_N)) = 0
    # y_{N-1} (A_N + C_N) + y_N (B_N + 2h C_N) = -2h C_N

    PN = p_func(X[N])
    QN = q_func(X[N])
    RN = r_func(X[N])

    AN = PN - QN * h / 2.0
    BN = -2 * PN + RN * h ** 2
    CN = PN + QN * h / 2.0

    Alpha[N] = AN + CN
    Beta[N] = BN + 2 * h * CN
    Phi[N] = -2 * h * CN

    # --- Метод прогонки (Thomas Algorithm) ---
    # Прямой ход
    P_coeff = [0.0] * (N + 1)
    Q_coeff = [0.0] * (N + 1)

    # Для i=0: Beta[0]*y0 + Gamma[0]*y1 = Phi[0]
    # y0 = -Gamma[0]/Beta[0] * y1 + Phi[0]/Beta[0]
    P_coeff[0] = -Gamma[0] / Beta[0]
    Q_coeff[0] = Phi[0] / Beta[0]

    for i in range(1, N + 1):
        denom = Beta[i] + Alpha[i] * P_coeff[i - 1]
        if i < N:
            P_coeff[i] = -Gamma[i] / denom
        Q_coeff[i] = (Phi[i] - Alpha[i] * Q_coeff[i - 1]) / denom

    # Обратный ход
    Y = [0.0] * (N + 1)
    Y[N] = Q_coeff[N]

    for i in range(N - 1, -1, -1):
        Y[i] = P_coeff[i] * Y[i + 1] + Q_coeff[i]

    return X, Y


# ---------------------------------------------------------
# Оценка погрешности и вывод
# ---------------------------------------------------------

def runge_romberg_error(space: Tuple[float, float], h: float,
                        solutions: Tuple[List[float], List[float], List[float], List[float]],
                        p: int = 2) -> List[Tuple[float, float, float]]:
    """
    Оценка погрешности методом Рунге-Ромберга.
    p - порядок точности метода (для КРМ и Стрельбы с РК4 можно брать p=2 или p=4 в зависимости от условий).
    В данном случае для КРМ p=2.
    """
    x_h, y_h, x_2h, y_2h = solutions
    results = []

    for i, x_val in enumerate(x_2h):
        # Находим соответствие в сетке с шагом h
        idx_h = -1
        for k, val in enumerate(x_h):
            if abs(val - x_val) < 1e-9:
                idx_h = k
                break

        if idx_h != -1:
            y1 = y_h[idx_h]
            y2 = y_2h[i]
            # Формула Рунге
            err = (y1 - y2) / (2 ** p - 1)
            results.append((x_val, y1, err))

    return results


def print_results(X: List[float], Y: List[float],
                  errors: List[Tuple[float, float, float]],
                  title: str):
    err_dict = {res[0]: res[2] for res in errors}

    print(f"\n{title:^100}")
    print("=" * 115)
    print(f"{'x':<8} {'y_approx':<15} {'y_exact':<15} {'Abs Error':<15} {'R_h (Runge)':<15}")
    print("-" * 115)

    for x, y_calc in zip(X, Y):
        y_ex = exact_solution(x)
        abs_err = abs(y_calc - y_ex)

        # Ищем ошибку по Рунге для этой точки
        rh_val = "N/A"
        for k in err_dict:
            if abs(k - x) < 1e-9:
                rh_val = f"{err_dict[k]:.6e}"
                break

        print(f"{x:<8.2f} {y_calc:<15.8f} {y_ex:<15.8f} {abs_err:<15.6e} {rh_val:<15}")
    print()


def main():
    space = (0.0, 1.0)
    h = 0.1

    print("Решение краевой задачи: (e^x + 1)y'' - 2y' - e^x y = 0")
    print(f"Отрезок: {space}, Шаг: {h}")

    # --- Метод Стрельбы ---
    # Решаем с шагом h
    X_shoot_h, Y_shoot_h = shooting_method(space, h)
    # Решаем с шагом 2h (для оценки ошибки)
    X_shoot_2h, Y_shoot_2h = shooting_method(space, 2 * h)

    # Оценка погрешности (RK4 имеет порядок 4, но общий порядок стрельбы зависит от нахождения корня,
    # однако само численное интегрирование O(h^4), давайте возьмем p=4 для чистоты интегрирования)
    err_shoot = runge_romberg_error(space, h, (X_shoot_h, Y_shoot_h, X_shoot_2h, Y_shoot_2h), p=4)
    print_results(X_shoot_h, Y_shoot_h, err_shoot, "Метод Стрельбы (Shooting Method)")

    # --- Конечно-разностный метод (КРМ) ---
    # Решаем с шагом h
    X_fdm_h, Y_fdm_h = finite_difference_method(space, h)
    # Решаем с шагом 2h
    X_fdm_2h, Y_fdm_2h = finite_difference_method(space, 2 * h)

    # Оценка погрешности (КРМ имеет порядок точности O(h^2))
    err_fdm = runge_romberg_error(space, h, (X_fdm_h, Y_fdm_h, X_fdm_2h, Y_fdm_2h), p=2)
    print_results(X_fdm_h, Y_fdm_h, err_fdm, "Конечно-разностный метод (FDM)")


if __name__ == "__main__":
    main()