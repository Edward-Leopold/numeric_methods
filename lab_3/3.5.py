from typing import Callable
from math import sqrt


def f(x: float) -> float:
    """Подынтегральная функция"""
    return sqrt(x) / (4 + 3 * x)


def rectangles_method(func: Callable, a: float, b: float, h: float) -> float:
    """Метод средних прямоугольников"""
    total = 0.0
    x = a + h / 2
    while x < b:
        total += func(x) * h
        x += h
    return total


def trapezoids_method(func: Callable, a: float, b: float, h: float) -> float:
    """Метод трапеций"""
    total = (func(a) + func(b)) / 2
    x = a + h
    while x < b:
        total += func(x)
        x += h
    return total * h


def simpson_method(func: Callable, a: float, b: float, h: float) -> float:
    """Метод Симпсона"""
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
    """Уточнение по Рунге-Ромбергу"""
    return I_h2 + (I_h2 - I_h) / (2 ** p - 1)


def main():
    a = 0
    b = 2
    h1 = 0.5
    h2 = 0.25

    print("Вычисление интеграла ∫(√x/(4+3x))dx от 0 до 2")
    print(f"Шаги: h1 = {h1}, h2 = {h2}")
    print("=" * 70)

    # Метод прямоугольников
    I1_rect = rectangles_method(f, a, b, h1)
    I2_rect = rectangles_method(f, a, b, h2)
    refined_rect = runge_romberg_refinement(I1_rect, I2_rect, 2)

    # Метод трапеций
    I1_trap = trapezoids_method(f, a, b, h1)
    I2_trap = trapezoids_method(f, a, b, h2)
    refined_trap = runge_romberg_refinement(I1_trap, I2_trap, 2)

    # Метод Симпсона
    I1_simp = simpson_method(f, a, b, h1)
    I2_simp = simpson_method(f, a, b, h2)
    refined_simp = runge_romberg_refinement(I1_simp, I2_simp, 4)

    print(f"{'Метод':<20} | h={h1} | h={h2} | Уточненное")
    print("-" * 70)
    print(f"{'Прямоугольники':<20} | {I1_rect:.6f} | {I2_rect:.6f} | {refined_rect:.6f}")
    print(f"{'Трапеции':<20} | {I1_trap:.6f} | {I2_trap:.6f} | {refined_trap:.6f}")
    print(f"{'Симпсон':<20} | {I1_simp:.6f} | {I2_simp:.6f} | {refined_simp:.6f}")


if __name__ == '__main__':
    main()