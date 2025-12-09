from typing import Callable
from math import sqrt


def f(x: float) -> float:
    return x / (3*x + 4) ** 2

def rectangles_method(func: Callable, a: float, b: float, h: float) -> float:
    total = 0.0
    x = a + h / 2
    while x < b:
        total += func(x) * h
        x += h
    return total


def trapezoids_method(func: Callable, a: float, b: float, h: float) -> float:
    total = (func(a) + func(b)) / 2
    x = a + h
    while x < b:
        total += func(x)
        x += h
    return total * h


def simpson_method(func: Callable, a: float, b: float, h: float) -> float:
    n = int((b - a) / h)
    if n % 2 != 0:
        n -= 1

    total = func(a) + func(b)
    for i in range(1, n):
        x = a + i * h
        if i % 2 == 0:
            total += 2 * func(x)
        else:
            total += 4 * func(x)

    return total * h / 3


def runge_romberg_refinement(I_h: float, I_h2: float, p: int) -> float:
    return I_h2 + (I_h2 - I_h) / (2 ** p - 1)


def main():
    a = -1
    b = 1
    h1 = 0.5
    h2 = 0.25

    # Точное значение интеграла (вычислено аналитически)
    exact_value = -0.16474

    print(f"Шаги: h1 = {h1}, h2 = {h2}")
    print(f"Точное значение: {exact_value:.6f}")
    print()

    # Метод прямоугольников
    I1_rect = rectangles_method(f, a, b, h1)
    I2_rect = rectangles_method(f, a, b, h2)
    refined_rect = runge_romberg_refinement(I1_rect, I2_rect, 2)
    error_rect = abs(refined_rect - exact_value)

    # Метод трапеций
    I1_trap = trapezoids_method(f, a, b, h1)
    I2_trap = trapezoids_method(f, a, b, h2)
    refined_trap = runge_romberg_refinement(I1_trap, I2_trap, 2)
    error_trap = abs(refined_trap - exact_value)

    # Метод Симпсона
    I1_simp = simpson_method(f, a, b, h1)
    I2_simp = simpson_method(f, a, b, h2)
    refined_simp = runge_romberg_refinement(I1_simp, I2_simp, 4)
    error_simp = abs(refined_simp - exact_value)

    print(f"{'Метод':<20} | {'h='+str(h1):<10} | {'h='+str(h2):<10} | {'Уточненное':<10} | {'Погрешность':<12}")
    print("-" * 75)
    print(f"{'Прямоугольники':<20} | {I1_rect:>10.6f} | {I2_rect:>10.6f} | {refined_rect:>10.6f} | {error_rect:>12.2e}")
    print(f"{'Трапеции':<20} | {I1_trap:>10.6f} | {I2_trap:>10.6f} | {refined_trap:>10.6f} | {error_trap:>12.2e}")
    print(f"{'Симпсон':<20} | {I1_simp:>10.6f} | {I2_simp:>10.6f} | {refined_simp:>10.6f} | {error_simp:>12.2e}")


if __name__ == '__main__':
    main()