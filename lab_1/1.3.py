from typing import List
from common.matrix import Matrix


def norm_matrix(matrix: Matrix) -> float:
    norm = 0.0
    for i in range(matrix.rows):
        row_sum = sum(abs(matrix.matrix[i][j]) for j in range(matrix.cols))
        norm = max(norm, row_sum)
    return norm


def norm_vector(x: List[float], previous_x: List[float]) -> float:
    return max(abs(x[i] - previous_x[i]) for i in range(len(x)))


def rearrange_matrix(A: Matrix, b: List[float]) -> tuple:
    n = A.rows
    A_data = [row[:] for row in A.matrix]  
    b_new = b[:]  

    for i in range(n):
        if abs(A_data[i][i]) < 1e-10:
            max_row = i
            max_val = abs(A_data[i][i])
            for k in range(i + 1, n):
                if abs(A_data[k][i]) > max_val:
                    max_val = abs(A_data[k][i])
                    max_row = k

            if max_row != i:
                A_data[i], A_data[max_row] = A_data[max_row], A_data[i]
                b_new[i], b_new[max_row] = b_new[max_row], b_new[i]
                print(f"Переставлены строки {i} и {max_row}")

    return Matrix(A_data), b_new

def simple_iteration_method(A: Matrix, b: List[float], epsilon: float = 1e-6) -> List[float]:
    n = A.rows

    if A.rows != A.cols:
        raise ValueError("Матрица должна быть квадратной")
    if A.determinant() == 0:
        raise ValueError("Матрица вырождена")

    A, b = rearrange_matrix(A, b)

    alpha_data = []
    beta_vector = []
    for i in range(n):
        beta_i = b[i] / A.matrix[i][i]
        beta_vector.append(beta_i)

        alpha_row = []
        for j in range(n):
            if i == j:
                alpha_row.append(0.0)
            else:
                alpha_row.append(-A.matrix[i][j] / A.matrix[i][i])
        alpha_data.append(alpha_row)
    alpha_matrix = Matrix(alpha_data)

    matrix_norm = norm_matrix(alpha_matrix)
    if matrix_norm >= 1:
        print(f"Норма матрицы alpha = {matrix_norm} >= 1, достаточное условие сходимости не выполнено")

    # Начальное приближение
    x = beta_vector.copy()
    previous_x = [0.0] * n

    coefficient = matrix_norm / (1 - matrix_norm) if matrix_norm < 1 else matrix_norm
    print(f"Начало итераций:")
    print(f"Норма матрицы alpha: {matrix_norm}")
    print(f"Коэффициент для условия остановки: {coefficient}")
    print("-" * 50)

    iteration = 0
    while True:
        previous_x = x.copy()
        x_new = [0.0] * n

        for i in range(n):
            sum_alpha_x = 0.0
            for j in range(n):
                sum_alpha_x += alpha_matrix.matrix[i][j] * previous_x[j]
            x_new[i] = beta_vector[i] + sum_alpha_x

        x = x_new
        iteration += 1

        current_norm = norm_vector(x, previous_x)
        if coefficient * current_norm < epsilon:
            print(f"Решение найдено за {iteration} итераций")
            print(f"Точность: {current_norm}")
            break

        print(f"Итерация {iteration}, норма разности: {current_norm}")

    return x


def seidel_method(A: Matrix, b: List[float], epsilon: float = 1e-6) -> List[float]:
    n = A.rows

    if A.rows != A.cols:
        raise ValueError("Матрица должна быть квадратной")
    if A.determinant() == 0:
        raise ValueError("Матрица вырождена")

    A, b = rearrange_matrix(A, b)

    x = [0.0] * n
    previous_x = [0.0] * n

    print(f"Метод Зейделя:")
    print(f"Начало итераций:")
    print("-" * 50)

    iteration = 0
    while True:
        previous_x = x.copy()

        for i in range(n):
            sum1 = 0.0
            sum2 = 0.0

            for j in range(i):
                sum1 += A.matrix[i][j] * x[j]

            for j in range(i + 1, n):
                sum2 += A.matrix[i][j] * previous_x[j]

            x[i] = (b[i] - sum1 - sum2) / A.matrix[i][i]

        iteration += 1

        current_norm = norm_vector(x, previous_x)
        if current_norm < epsilon:
            print(f"Решение найдено за {iteration} итераций")
            print(f"Точность: {current_norm}")
            break

        print(f"Итерация {iteration}, норма разности: {current_norm}")

    return x

def main():
    A1 = Matrix([
        [-22, -2, -6, 6],
        [3, -17, -3, 7],
        [2, 6, -17, 5],
        [-1, -8, 8, 23]
    ])
    b1 = [96, -26, 35, -234]

    try:
        x1 = simple_iteration_method(A1, b1, 0.0001)
        print(f"Решение: {x1}")

    except Exception as e:
        print(f"Ошибка: {e}")
    print(f"\n {50 * '='} \n")
    try:
        x1 = seidel_method(A1, b1, 0.0001)
        print(f"Решение: {x1}")

    except Exception as e:
        print(f"Ошибка: {e}")


if __name__ == '__main__':
    main()