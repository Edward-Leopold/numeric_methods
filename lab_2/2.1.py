import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return (x * x * x) - 2 * (x * x) - 10 * x + 15


def df(x):
    return 3 * (x ** 2) - 4 * x - 10


def d2f(x):
    return 6 * x - 4


def phi(x):
    return (x**3 - 2*x**2 + 15) / 10

def phi_dev(x):
    return (3*x**2 - 4*x) / 10


def newton_check(a, b, eps):
    if f(a) * f(b) > 0:
        return False
    return True


def iterations_check(a, b, eps):
    for x in np.arange(a, b, 0.001):
        if abs(phi_dev(x)) > 1:
            return False
    return True


def newton(a, b, eps, max_iter=1000):
    if (f(a) * d2f(a)) > 0:
        x = a
    else:
        x = b
    for i in range(max_iter):
        x_next = x - f(x) / df(x)
        if abs(x_next - x) < eps:
            return x_next, i
        x = x_next
    return x_next, i


def iterations(a, b, eps, max_iter=1000):
    x = (a + b) / 2
    q = 0
    for x in np.arange(a, b, eps):
        if abs(phi_dev(x)) > q:
            q = abs(phi_dev(x))
    for i in range(max_iter):
        x_next = phi(x)
        if abs(x_next - x) * (q / (1 - q)) < eps:
            return x_next, i, q
        x = x_next
    return x_next, i, q


def graph_1():
    x = np.linspace(-2, 5, 100)
    y = f(x)
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, label='f(x) = x³ - 2x² - 10x + 15')
    plt.axhline(0, color='black', linewidth=2)
    plt.axvline(0, color='black', linewidth=2)
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.xticks(np.arange(-2, 5.5, 0.5))
    plt.show()


def graph_2(a, b):
    epsilons = [10 ** (-i) for i in range(1, 7)]
    eps_labels = [f"1e-{i}" for i in range(1, 7)]

    iterations_newton = []
    iterations_simple = []

    for eps in epsilons:
        _, iters_n = newton(a, b, eps)
        iterations_newton.append(iters_n)

        _, iters_s = iterations(a, b, eps)
        iterations_simple.append(iters_s)

    plt.figure(figsize=(10, 5))
    plt.plot(eps_labels, iterations_newton, marker='o', linestyle='-', color='green', label='Метод Ньютона')
    plt.plot(eps_labels, iterations_simple, marker='s', linestyle='-', color='blue', label='Метод простых итераций')

    plt.xlabel('Точность ε')
    plt.ylabel('Число итераций')
    plt.title('Зависимость числа итераций от точности ε')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)

    ymin = min(iterations_newton + iterations_simple)
    ymax = max(iterations_newton + iterations_simple)
    plt.yticks(range(ymin, ymax + 1))

    plt.tight_layout()
    plt.show()


def main():
    graph_1()
    a, b = map(float, input("Введите a и b через пробел: ").split())
    eps = float(input("Введите погрешность: "))

    print("Метод Ньютона:")
    if newton_check(a, b, eps):
        x, i = newton(a, b, eps)
        print("Итераций:", i)
        print("Решение: x =", x)
    else:
        print("Выбранный интервал не подходит для решения методом Ньютона")

    print("Метод простых итераций:")
    if iterations_check(a, b, eps):
        x, i, q = iterations(a, b, eps)
        print("Итераций:", i)
        print("Решение: x =", x)
        print("q =", q)
    else:
        print("Выбранный интервал не подходит для решения методом простых итераций")

    # graph_2(a, b)


if __name__ == "__main__":
    main()