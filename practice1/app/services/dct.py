import numpy as np

class DCT:
    def __init__(self, data):
        self.data = data
        self.N = data.shape[0]

    def dct(self):
        N = self.N
        transformed = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                sum_value = 0
                for x in range(N):
                    for y in range(N):
                        sum_value += self.data[x, y] * \
                                     np.cos((2 * x + 1) * u * np.pi / (2 * N)) * \
                                     np.cos((2 * y + 1) * v * np.pi / (2 * N))
                alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                alpha_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
                transformed[u, v] = alpha_u * alpha_v * sum_value

        self.transformed_data = transformed
        return transformed

    def idct(self):
        N = self.N
        reconstructed = np.zeros((N, N))
        for x in range(N):
            for y in range(N):
                sum_value = 0
                for u in range(N):
                    for v in range(N):
                        alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                        alpha_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
                        sum_value += alpha_u * alpha_v * \
                                     self.transformed_data[u, v] * \
                                     np.cos((2 * x + 1) * u * np.pi / (2 * N)) * \
                                     np.cos((2 * y + 1) * v * np.pi / (2 * N))
                reconstructed[x, y] = sum_value

        return reconstructed
