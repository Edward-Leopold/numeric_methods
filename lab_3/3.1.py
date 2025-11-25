import math
from typing import List, Callable


def f(x: float) -> float:
    return math.tan(x) + x


def get_y_values(func: Callable[[float], float], X_values: List[float]) -> List[float]:
    return [func(x) for x in X_values]


def lagrange_polynom(x: float, X_values: List[float], Y_values: List[float]) -> float:
    n = len(X_values)
    res = 0.0

    for i in range(n):
        term = Y_values[i]
        for j in range(n):
            if i != j:
                term *= (x - X_values[j]) / (X_values[i] - X_values[j])
        res += term

    return res


def divided_diff(func: Callable[[float], float], X_values: List[float]) -> float:
    if len(X_values) == 1:
        return func(X_values[0])
    elif len(X_values) == 2:
        return (func(X_values[1]) - func(X_values[0])) / (X_values[1] - X_values[0])

    return (divided_diff(func, X_values[1:]) - divided_diff(func, X_values[:-1])) / (X_values[-1] - X_values[0])


def newton_polynom(x: float, func: Callable[[float], float], diff_func: Callable[[Callable[[float], float], List[float]], float], X_values: List[float]) -> float:
    n = len(X_values)
    res = 0.0

    for i in range(n):
        term = diff_func(func, X_values[:i + 1])
        for j in range(i):
            term *= (x - X_values[j])
        res += term

    return res


def fault(x: float, func: Callable[[float], float], possible_value: float) -> float:
    return abs(func(x) - possible_value)


def main():
    X_a = [0, math.pi / 8, 2 * math.pi / 8, 3 * math.pi / 8]
    X_b = [0, math.pi / 8, math.pi / 3, 3 * math.pi / 8]
    X_star = 3 * math.pi / 16

    print("Интерполяция функции y = tg(x) + x")
    print("=" * 60)
    print(f"Точка интерполяции X* = 3π/16 = {X_star:.6f}")
    print(f"Точное значение в X*: f(X*) = {f(X_star):.6f}")
    print()

    print("а)")
    print("0, π/8, 2π/8, 3π/8")

    Y_a = get_y_values(f, X_a)

    print("Узлы интерполяции и значения функции:")
    for i, (x, y) in enumerate(zip(X_a, Y_a)):
        print(f"x{i} = {x:.6f}, y{i} = {y:.6f}")

    lagrange_val_a = lagrange_polynom(X_star, X_a, Y_a)
    newton_val_a = newton_polynom(X_star, f, divided_diff, X_a)
    print(f"\nРезультаты интерполяции для случая а):")
    print(f"Метод Лагранжа: {lagrange_val_a:.8f}")
    print(f"Погрешность Лагранжа: {fault(X_star, f, lagrange_val_a):.2e}")
    print(f"Метод Ньютона:  {newton_val_a:.8f}")
    print(f"Погрешность Ньютона:  {fault(X_star, f, newton_val_a):.2e}")

    print("\n" + "=" * 60)
    print("б)")
    print("0, π/8, π/3, 3π/8")

    Y_b = get_y_values(f, X_b)

    print("Узлы интерполяции и значения функции:")
    for i, (x, y) in enumerate(zip(X_b, Y_b)):
        print(f"x{i} = {x:.6f}, y{i} = {y:.6f}")

    lagrange_val_b = lagrange_polynom(X_star, X_b, Y_b)
    newton_val_b = newton_polynom(X_star, f, divided_diff, X_b)
    print(f"\nРезультаты интерполяции для случая б):")
    print(f"Метод Лагранжа: {lagrange_val_b:.8f}")
    print(f"Погрешность Лагранжа: {fault(X_star, f, lagrange_val_b):.2e}")
    print(f"Метод Ньютона:  {newton_val_b:.8f}")
    print(f"Погрешность Ньютона:  {fault(X_star, f, newton_val_b):.2e}")




if __name__ == '__main__':
    main()