from typing import List
from common.matrix import Matrix

def round_matrix(matrix: Matrix, decimals: int = 6) -> Matrix:
    rounded_data = [[round(x, decimals) for x in row] for row in matrix.matrix]
    return Matrix(rounded_data)
def lup_decomposition(A: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    n = A.rows
    C_data = [row[:] for row in A.matrix]
    C = Matrix(C_data)
    P = Matrix.identity(n)

    def swap_rows(matrix: Matrix, i: int, j: int):
        matrix.matrix[i], matrix.matrix[j] = matrix.matrix[j], matrix.matrix[i]

    for i in range(n):
        pivot_value = 0.0
        pivot = -1
        for row in range(i, n):
            if abs(C.matrix[row][i]) > pivot_value:
                pivot_value = abs(C.matrix[row][i])
                pivot = row

        if pivot_value == 0:
            raise ValueError("Матрица вырождена")

        if pivot != i:
            swap_rows(P, pivot, i)
            swap_rows(C, pivot, i)

        for j in range(i + 1, n):
            C.matrix[j][i] /= C.matrix[i][i]
            for k in range(i + 1, n):
                C.matrix[j][k] -= C.matrix[j][i] * C.matrix[i][k]

    # C = L + U - E
    L_data = [[0.0 for _ in range(n)] for _ in range(n)]
    U_data = [[0.0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                L_data[i][j] = 1.0
                U_data[i][j] = C.matrix[i][j]
            elif i > j:
                L_data[i][j] = C.matrix[i][j]
            else:
                U_data[i][j] = C.matrix[i][j]

    L = Matrix(L_data)
    U = Matrix(U_data)

    return L, U, P


def solve_triangular_lower(L: Matrix, b: List[float]) -> List[float]:
    n = L.rows
    y = [0.0] * n

    for i in range(n):
        total = 0.0
        for j in range(i):
            total += L.matrix[i][j] * y[j]
        y[i] = (b[i] - total) / L.matrix[i][i]

    return y

def solve_triangular_upper(U: Matrix, y: List[float]) -> List[float]:
    n = U.rows
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        total = 0.0
        for j in range(i + 1, n):
            total += U.matrix[i][j] * x[j]
        x[i] = (y[i] - total) / U.matrix[i][i]

    return x


def inverse_via_lup_internal(A: Matrix, L: Matrix, U: Matrix, P: Matrix) -> Matrix:
    n = A.rows
    I = Matrix.identity(n)
    inv_data = []

    for j in range(n):
        I_column = [I.matrix[i][j] for i in range(n)]
        PI_column = [0.0] * n
        for i in range(P.rows):
            for k in range(P.cols):
                if P.matrix[i][k] == 1.0:
                    PI_column[i] = I_column[k]
                    break
        Y = solve_triangular_lower(L, PI_column)
        X_column = solve_triangular_upper(U, Y)

        inv_data.append(X_column)
    inv_data_transposed = [[inv_data[j][i] for j in range(n)] for i in range(n)]
    return Matrix(inv_data_transposed)

def solve_system_lup(A: Matrix, b: List[float]) -> List[float]:
    L, U, P = lup_decomposition(A)

    print("Проверка LU = A:")
    print(L * U)
    print("\n")
    print(P * A)

    det = L.determinant() * U.determinant()
    print(f"\nОпределитель матрицы A: {det}")

    det = A.determinant()
    print(f"\nОпределитель матрицы A: {det}")

    if A.is_invertible():
        A_inv = A.inverse()
        rounded_inv_data = [[round(x, 6) for x in row] for row in A_inv.matrix]
        rounded_inv = Matrix(rounded_inv_data)
        print("\nОбратная матрица A^(-1) (6 знаков):")
        print(rounded_inv)

        A_inv_lup = inverse_via_lup_internal(A, L, U, P)
        rounded_inv_lup = round_matrix(A_inv_lup, 6)
        print("\nОбратная матрица A^(-1) (LUP, AX = I):")
        print(rounded_inv_lup)

        print("\n Проверка A * A^(-1) = E")
        print(round_matrix(A * A_inv_lup , 6))
    else:
        print("\nМатрица A вырождена")

    pb = [0.0] * len(b)
    for i in range(P.rows):
        for j in range(P.cols):
            if P.matrix[i][j] == 1.0:
                pb[i] = b[j]
                break

    y = solve_triangular_lower(L, pb)
    x = solve_triangular_upper(U, y)

    return x

def inverse_via_lup_internal(A: Matrix, L: Matrix, U: Matrix, P: Matrix) -> Matrix:
    n = A.rows
    I = Matrix.identity(n)
    inv_data = []

    for j in range(n):
        I_column = [I.matrix[i][j] for i in range(n)]
        PI_column = [0.0] * n
        for i in range(P.rows):
            for k in range(P.cols):
                if P.matrix[i][k] == 1.0:
                    PI_column[i] = I_column[k]
                    break
        Y = solve_triangular_lower(L, PI_column)
        X_column = solve_triangular_upper(U, Y)

        inv_data.append(X_column)
    inv_data_transposed = [[inv_data[j][i] for j in range(n)] for i in range(n)]
    return Matrix(inv_data_transposed)

def main():
    m = Matrix([
        [-1, -3, -4, 0, -3],
        [3, 7, -8, 3, 30],
        [1, -6, 2, 5, -90],
        [-8,  -4, -1, -1, 12]
    ])

    A_data = []
    b = []

    for i in range(m.rows):
        A_data.append(m.matrix[i][:-1])
        b.append(m.matrix[i][-1])

    A = Matrix(A_data)

    print("Матрица коэффициентов A:")
    print(A)
    print(f"\nВектор правой части b: {b}")

    try:
        solution = solve_system_lup(A, b)

        rounded_solution = [round(x, 10) for x in solution]
        print(f"\nРешение СЛАУ: {rounded_solution}")

    except ValueError as e:
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()