import numpy as np

class DWT:
    def __init__(self, data):
        self.data = data
        self.N = data.shape[0]  # Assumes a square matrix

    def dwt(self):
        N = self.N
        approx = np.zeros((N // 2, N // 2))
        horiz = np.zeros((N // 2, N // 2))
        vert = np.zeros((N // 2, N // 2))
        diag = np.zeros((N // 2, N // 2))
        
        for i in range(0, N, 2):
            for j in range(0, N, 2):
                block = self.data[i:i+2, j:j+2]
                avg = (block[0, 0] + block[0, 1] + block[1, 0] + block[1, 1]) / 4
                horiz_detail = (block[0, 0] - block[0, 1] + block[1, 0] - block[1, 1]) / 4
                vert_detail = (block[0, 0] + block[0, 1] - block[1, 0] - block[1, 1]) / 4
                diag_detail = (block[0, 0] - block[0, 1] - block[1, 0] + block[1, 1]) / 4
                
                approx[i//2, j//2] = avg
                horiz[i//2, j//2] = horiz_detail
                vert[i//2, j//2] = vert_detail
                diag[i//2, j//2] = diag_detail

        self.coeffs = (approx, horiz, vert, diag)
        return self.coeffs

    def idwt(self):
        approx, horiz, vert, diag = self.coeffs
        N = self.N
        reconstructed = np.zeros((N, N))
        
        for i in range(0, N, 2):
            for j in range(0, N, 2):
                avg = approx[i//2, j//2]
                horiz_detail = horiz[i//2, j//2]
                vert_detail = vert[i//2, j//2]
                diag_detail = diag[i//2, j//2]
                
                reconstructed[i, j]     = avg + horiz_detail + vert_detail + diag_detail
                reconstructed[i, j+1]   = avg - horiz_detail + vert_detail - diag_detail
                reconstructed[i+1, j]   = avg + horiz_detail - vert_detail - diag_detail
                reconstructed[i+1, j+1] = avg - horiz_detail - vert_detail + diag_detail

        return reconstructed
