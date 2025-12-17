from typing import List

class Matrix:
    def __init__(self, matrix: List[List[int | float]]):
        self.matrix: List[List[int | float]] = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0]) if matrix and matrix[0] else 0

    @staticmethod
    def identity(n: int) -> 'Matrix':
        data = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        return Matrix(data)

    def determinant(self) -> float:
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")

        if self.rows == 1:
            return self.matrix[0][0]
        elif self.rows == 2:
            return self.matrix[0][0] * self.matrix[1][1] - self.matrix[0][1] * self.matrix[1][0]
        else:
            result = 0
            for k in range(self.rows):
                minor_matrix = []
                for i in range(1, self.rows):
                    row = []
                    for j in range(self.cols):
                        if j != k:
                            row.append(self.matrix[i][j])
                    minor_matrix.append(row)

                minor = Matrix(minor_matrix)
                coefficient = self.matrix[0][k]
                result += (-1) ** k * coefficient * minor.determinant()

            return result

    def multiply(self, other: 'Matrix') -> 'Matrix':
        if self.cols != other.rows:
            raise ValueError('Incorrect matrix sizes')

        result_data = [[0 for _ in range(other.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                total = 0
                for r in range(self.cols):
                    total += self.matrix[i][r] * other.matrix[r][j]
                result_data[i][j] = total

        return Matrix(result_data)

    def __mul__(self, other: 'Matrix') -> 'Matrix':
        return self.multiply(other)

    def transpose(self) -> 'Matrix':
        if self.rows == 0 or self.cols == 0:
            return Matrix([[]])

        transposed = [[0] * self.rows for _ in range(self.cols)]

        for i in range(self.rows):
            for j in range(self.cols):
                transposed[j][i] = self.matrix[i][j]

        return Matrix(transposed)

    def is_invertible(self) -> bool:
        if self.rows != self.cols:
            return False
        return self.determinant() != 0

    def inverse(self) -> 'Matrix':
        if not self.is_invertible():
            raise ValueError("Матрица вырождена и не имеет обратной")

        n = self.rows

        augmented = []
        for i in range(n):
            row = self.matrix[i][:]
            row.extend([1.0 if j == i else 0.0 for j in range(n)])
            augmented.append(row)

        for i in range(n):
            pivot_row = i
            for r in range(i + 1, n):
                if abs(augmented[r][i]) > abs(augmented[pivot_row][i]):
                    pivot_row = r

            augmented[i], augmented[pivot_row] = augmented[pivot_row], augmented[i]
            pivot_val = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= pivot_val

            for r in range(i + 1, n):
                factor = augmented[r][i]
                for c in range(2 * n):
                    augmented[r][c] -= factor * augmented[i][c]

        for i in range(n - 1, -1, -1):
            for r in range(i - 1, -1, -1):
                factor = augmented[r][i]
                for c in range(2 * n):
                    augmented[r][c] -= factor * augmented[i][c]

        inverse_data = []
        for i in range(n):
            inverse_data.append(augmented[i][n:])

        return Matrix(inverse_data)

    @staticmethod
    def calculate_norm(matrix: List[List[int | float]]) -> int | float:
        return max(map(lambda row: sum(map(abs, row)), matrix))

    def __str__(self) -> str:
        return '\n'.join([' '.join(map(str, row)) for row in self.matrix])