from typing import List, Tuple


def find_interval(X: List[float], Y: List[float], x_point: float) -> Tuple[List[float], List[float]]:
    """
    Находит интервал для вычисления производных
    """
    if x_point < X[0] or x_point > X[-1]:
        raise ValueError('Точка вне диапазона данных')

    for i in range(len(X) - 2):
        if X[i] <= x_point <= X[i + 1]:
            return X[i:i + 3], Y[i:i + 3]

    return X[-3:], Y[-3:]


def calculate_first_derivative(X: List[float], Y: List[float], x_point: float) -> float:
    """
    Вычисляет первую производную
    """
    x1, x2, x3 = X
    y1, y2, y3 = Y

    term1 = (y2 - y1) / (x2 - x1)
    term2 = ((y3 - y2) / (x3 - x2) - term1) / (x3 - x1)

    return term1 + term2 * (2 * x_point - x1 - x2)


def calculate_second_derivative(X: List[float], Y: List[float]) -> float:
    """
    Вычисляет вторую производную
    """
    x1, x2, x3 = X
    y1, y2, y3 = Y

    term1 = (y2 - y1) / (x2 - x1)
    term2 = (y3 - y2) / (x3 - x2)

    return 2 * (term2 - term1) / (x3 - x1)


def main():
    # Данные 14 варианта
    X = [1.0, 2.0, 3.0, 4.0, 5.0]
    Y = [1.0, 2.6931, 4.0986, 5.3863, 6.6094]
    x_star = 3.0

    # Находим интервал и вычисляем производные
    X_interval, Y_interval = find_interval(X, Y, x_star)

    first_deriv = calculate_first_derivative(X_interval, Y_interval, x_star)
    second_deriv = calculate_second_derivative(X_interval, Y_interval)

    print(f"Точка X* = {x_star}")
    print(f"Используемый интервал: X = {X_interval}, Y = {Y_interval}")
    print(f"Первая производная: {first_deriv:.6f}")
    print(f"Вторая производная: {second_deriv:.6f}")


if __name__ == '__main__':
    main()