import math
from typing import List, Tuple, Set
from common.matrix import Matrix


def norm_matrix(matrix: Matrix) -> float:
    norm = 0.0
    for i in range(matrix.rows):
        row_sum = sum(abs(matrix.matrix[i][j]) for j in range(matrix.cols))
        norm = max(norm, row_sum)
    return norm


def norm_vector_diff(x: List[float], previous_x: List[float]) -> float:
    return max(abs(x[i] - previous_x[i]) for i in range(len(x)))


def norm_vector(vector: List[float]) -> float:
    return max(abs(x) for x in vector) if vector else 0.0

def norm_vector_euclidean(vector: List[float]) -> float:
    return math.sqrt(sum(x * x for x in vector)) if vector else 0.0

def householder_matrix(vector: List[float]) -> Tuple[Matrix, float]:
    n = len(vector)

    if all(abs(x) < 1e-15 for x in vector):
        return Matrix.identity(n), 0.0

    norm = norm_vector_euclidean(vector)
    v = vector.copy()
    if vector[0] >= 0:
        v[0] = vector[0] + norm
    else:
        v[0] = vector[0] - norm

    v_norm_squared = sum(x * x for x in v)
    beta = 2.0 / v_norm_squared if v_norm_squared > 1e-15 else 0.0

    H_data = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                H_data[i][j] = 1.0 - beta * v[i] * v[j]
            else:
                H_data[i][j] = -beta * v[i] * v[j]

    return Matrix(H_data), beta


def qr_decomposition_householder(A: Matrix) -> Tuple[Matrix, Matrix]:
    if A.rows == 0 or A.cols == 0:
        raise ValueError("Матрица не может быть пустой")

    m, n = A.rows, A.cols
    R = Matrix([row[:] for row in A.matrix])
    Q = Matrix.identity(m)

    for k in range(min(m, n)):
        x = [R.matrix[i][k] for i in range(k, m)]
        if norm_vector_euclidean(x) < 1e-10:
            continue

        H_k, beta = householder_matrix(x)
        H_full = Matrix.identity(m)
        for i in range(k, m):
            for j in range(k, m):
                H_full.matrix[i][j] = H_k.matrix[i - k][j - k]

        # R = H * R
        R = H_full.multiply(R)

        for i in range(k + 1, m):
            if abs(R.matrix[i][k]) < 1e-12:
                R.matrix[i][k] = 0.0

        # Q = Q * H
        Q = Q.multiply(H_full)

    return Q, R


def is_upper_triangular(A: Matrix, epsilon: float = 1e-6) -> bool:
    for i in range(1, A.rows):
        for j in range(min(i, A.cols)):
            if abs(A.matrix[i][j]) > epsilon:
                return False
    return True

def stop_condition(A: Matrix, accuracy: float, real_indexes: List[int]) -> bool:
    for i in range(A.rows):
        for j in range(A.cols):
            if (abs(A.matrix[i][j]) >= accuracy and i > j and j in real_indexes):
                return False
            if abs(A.matrix[i][j]) == float('inf'):
                return False
    return True

def qr_algorithm(A: Matrix, accuracy: float = 1e-10) -> Set[str]:
    if A.rows != A.cols:
        raise ValueError("Матрица должна быть квадратной")

    current_A = A
    eigenvalues = set()

    print(f"QR-алгоритм: матрица {A.rows}x{A.cols}, точность {accuracy}")
    print("Итерационный процесс:")
    mod_prev, mod_next = [], []
    iteration = 0
    while True:
        Q, R = qr_decomposition_householder(current_A)
        current_A = R.multiply(Q)
        eigenvalues.clear()
        real_indexes = []
        mod_next.clear()

        i = 0
        while i < current_A.rows:
            # СЗ для блока 2X2
            if i < current_A.rows - 1 and abs(current_A.matrix[i + 1][i]) > accuracy:
                a, b, c, d = (current_A.matrix[i][i], current_A.matrix[i][i + 1],
                              current_A.matrix[i + 1][i], current_A.matrix[i + 1][i + 1])

                trace = a + d
                det = a * d - b * c
                discriminant = trace ** 2 - 4 * det

                if discriminant >= 0:
                    # Вещественные СЗ
                    real_indexes.append(i)
                    lambda1 = (trace + math.sqrt(discriminant)) / 2
                    lambda2 = (trace - math.sqrt(discriminant)) / 2
                    eigenvalues.add(f'{lambda1:.10f}')
                    eigenvalues.add(f'{lambda2:.10f}')
                else:
                    # Комплексные СЗ
                    real_part = trace / 2
                    imag_part = math.sqrt(-discriminant) / 2
                    eigenvalues.add(f"{real_part:.10f}+{imag_part:.10f}j")
                    eigenvalues.add(f"{real_part:.10f}-{imag_part:.10f}j")
                    mod_next.append((real_part ** 2 + imag_part ** 2) ** 0.5)

                i += 2
            else:
                # Вещественное СЗ
                eigenvalues.add(f'{current_A.matrix[i][i]:.10f}')
                i += 1

        if (stop_condition(current_A, accuracy, real_indexes) and
                (all(abs(mod_prev[k] - mod_next[k]) < accuracy
                     for k in range(len(mod_prev))) if mod_prev else True)):
            print(f"Сходимость достигнута за {iteration + 1} итераций")
            break

        mod_prev = mod_next.copy()
        iteration += 1

    return eigenvalues

def main():
    print("\n" + "=" * 60)

    A = Matrix([
        [2, -4, 5],
        [-5, -2, -3],
        [1, -8, -3]
    ])
    print("Матрица A:")
    print(A)
    Q, R = qr_decomposition_householder(A)
    print("\nМатрица Q:")
    print(Q)
    print("\nМатрица R:")
    print(R)

    q_norm = norm_matrix(Q)
    r_norm = norm_matrix(R)
    print(f"\nНорма матрицы Q: {q_norm}")
    print(f"Норма матрицы R: {r_norm}")

    print("\nПроверка Q * R:")
    qr_product = Q.multiply(R)
    print(qr_product)

    print("\n" + "=" * 60)
    print("Нахождение собственных значений:")

    print(A)
    eigenvalues = qr_algorithm(A, 1e-8)
    print("Собственные значения:")
    for val in sorted(eigenvalues):
        print(f"  {val}")


if __name__ == "__main__":
    main()