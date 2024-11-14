class RGBtoYUV:
    def __init__(self, R, G, B):  # Corrected __init__ method
        self.R = R
        self.G = G
        self.B = B
    
    def conversor(self):
        Y = ((66 * self.R + 129 * self.G + 25 * self.B + 128) / 256) + 16
        U = ((-38 * self.R - 74 * self.G + 112 * self.B + 128) / 256) + 128
        V = ((112 * self.R - 94 * self.G - 18 * self.B + 128) / 256) + 128
        yuv = [Y, U, V]
        return yuv

# Test RGB to YUV conversion
rgb1 = RGBtoYUV(16, 128, 128)
yuv_values = rgb1.conversor()
print("YUV values:", yuv_values)

class YUVtoRGB:
    def __init__(self, Y, U, V):  # Corrected __init__ method
        self.Y = Y - 16
        self.U = U - 128
        self.V = V - 128
    
    def conversor(self):
        R = 1.164 * self.Y + 1.596 * self.V
        G = 1.164 * self.Y - 0.392 * self.U - 0.813 * self.V
        B = 1.164 * self.Y + 2.017 * self.U
        rgb = [R, G, B]
        return rgb

# Test YUV to RGB conversion
yuv1 = YUVtoRGB(97.625, 145.125, 79.5)
rgb_values = yuv1.conversor()
print("RGB values:", rgb_values)


#Exercise 3 resize an image using ffmpeg 
import subprocess

def resize_image(input_path, output_path, width, height):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-vf', f'scale={width}:{height}',
        output_path
    ]
    subprocess.run(command) 

input_image = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image.jpg'
output_image = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/output_320x240.jpg'
resize_image(input_image, output_image, 320, 240)


from PIL import Image
import numpy as np

def serpentine(image_path):
    
    img = Image.open(image_path).convert('L') #convert it to grayscale
    image_matrix = np.array(img)
    output = zig_zag_matrix_(image_matrix)
    output = np.clip(output, 0, 255).astype(np.uint8)
    output_bytes = bytearray(output)
    return output_bytes

def zig_zag_matrix_(mat):
    n = len(mat)
    m = len(mat[0])
    row = 0
    col = 0
    row_inc = False
    output = []  # List to store the zig-zag pattern values

    # Traverse the first half of the zig-zag pattern
    mn = min(m, n)
    for length in range(1, mn + 1):
        for i in range(length):
            output.append(mat[row][col])

            if i + 1 == length:
                break

            if row_inc:
                row += 1
                col -= 1
            else:
                row -= 1
                col += 1

        if length == mn:
            break

        if row_inc:
            row += 1
            row_inc = False
        else:
            col += 1
            row_inc = True

    # Adjust row and col for the second half of the matrix
    if row == 0:
        if col == m - 1:
            row += 1
        else:
            col += 1
        row_inc = True
    else:
        if row == n - 1:
            col += 1
        else:
            row += 1
        row_inc = False

    # Traverse the second half of the zig-zag pattern
    MAX = max(m, n) - 1
    for diag in range(MAX, 0, -1):
        length = mn if diag > mn else diag
        for i in range(length):
            output.append(mat[row][col])

            if i + 1 == length:
                break

            if row_inc:
                row += 1
                col -= 1
            else:
                col += 1
                row -= 1

        if row == 0 or col == m - 1:
            if col == m - 1:
                row += 1
            else:
                col += 1
            row_inc = True
        elif col == 0 or row == n - 1:
            if row == n - 1:
                col += 1
            else:
                row += 1
            row_inc = False

    return output

# Example usage
#image_path = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/landscape.jpg'  # Replace with the path to your JPEG image
image_path = "/Users/rogerarolaplanas/Documents/GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/landscape.jpg"
serpentine(image_path)


# 5) Use FFMPEG to transform the previous image into b/w. Do the hardest compression you can.

import subprocess

def convert_to_bw(input_path, output_path):
    subprocess.run([
        "ffmpeg", 
        "-i", input_path, 
        "-vf", "format=gray", 
        "-q:v", "31",  #31 is the lowest quality
        output_path
    ], check=True)

bw_image_input = "/Users/rogerarolaplanas/Documents/GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image.jpg"
bw_image_output = "/Users/rogerarolaplanas/Documents/GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image_bw.jpg"
convert_to_bw(bw_image_input, bw_image_output)


# 5) Create a method which applies a run-lenght encoding from a series of bytes given.

def printRLE(st):
 
    n = len(st)
    i = 0
    while i < n- 1:
 
        # Count occurrences of current character
        count = 1
        while (i < n - 1 and st[i] == st[i + 1]):
            count += 1
            i += 1
        i += 1
 
        # Print character and its count
        print(st[i - 1] + str(count), end = "")
 
#example found on the internet
st = "wwwwaaadexxxxxxywww"
printRLE(st)


# 6) Create a class which can convert, can decode (or both) an input using the DCT.
# Not necessary a JPG encoder or decoder. A class only about DCT is OK too

import numpy as np

class DCT:
    def __init__(self, data):
        self.data = data
        self.N = data.shape[0]

    # Following the formula for a 2D DCT for an NxN matrix
    def dct(self):
        N = self.N
        transformed = np.zeros((N, N))
        for u in range(N):
            for v in range(N):
                sum_value = 0
                for x in range(N):
                    for y in range(N):
                        sum_value += self.data[x, y] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                
                # Apply the scaling factors
                alpha_u = np.sqrt(1 / N) if u == 0 else np.sqrt(2 / N)
                alpha_v = np.sqrt(1 / N) if v == 0 else np.sqrt(2 / N)
                transformed[u, v] = alpha_u * alpha_v * sum_value

        self.transformed_data = transformed  # Store the transformed data for later use in IDCT
        return transformed

    # Following the formula for a 2D IDCT for an NxN matrix
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
                        sum_value += alpha_u * alpha_v * self.transformed_data[u, v] * np.cos((2 * x + 1) * u * np.pi / (2 * N)) * np.cos((2 * y + 1) * v * np.pi / (2 * N))
                
                reconstructed[x, y] = sum_value

        return reconstructed

# Example usage
if __name__ == "__main__":
    data = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 66, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ])

    dct_processor = DCT(data)
    transformed = dct_processor.dct()
    print("Computed DCT:\n", transformed)

    reconstructed = dct_processor.idct()
    print("Computed IDCT (reconstructed):\n", reconstructed)


# 7) Create a class which can convert, can decode (or both) an input using the DWT.
# Not necessary a JPEG2000 encoder or decoder. A class only about DWT is OK too

import numpy as np

class ManualDWT:
    def __init__(self, data):
        self.data = data
        self.N = data.shape[0]  # Assumes a square matrix

    def dwt(self):
        N = self.N
        approx = np.zeros((N // 2, N // 2))
        horiz = np.zeros((N // 2, N // 2))
        vert = np.zeros((N // 2, N // 2))
        diag = np.zeros((N // 2, N // 2))
        
        # Process each 2x2 block to calculate coefficients
        for i in range(0, N, 2):
            for j in range(0, N, 2):
                # Take a 2x2 block
                block = self.data[i:i+2, j:j+2]
                
                # Haar wavelet coefficients
                avg = (block[0, 0] + block[0, 1] + block[1, 0] + block[1, 1]) / 4  # Approximation
                horiz_detail = (block[0, 0] - block[0, 1] + block[1, 0] - block[1, 1]) / 4  # Horizontal
                vert_detail = (block[0, 0] + block[0, 1] - block[1, 0] - block[1, 1]) / 4  # Vertical
                diag_detail = (block[0, 0] - block[0, 1] - block[1, 0] + block[1, 1]) / 4  # Diagonal
                
                # Store the coefficients
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
        
        # Process each 2x2 block to reconstruct
        for i in range(0, N, 2):
            for j in range(0, N, 2):
                # Retrieve coefficients
                avg = approx[i//2, j//2]
                horiz_detail = horiz[i//2, j//2]
                vert_detail = vert[i//2, j//2]
                diag_detail = diag[i//2, j//2]
                
                # Reconstruct the 2x2 block
                reconstructed[i, j]     = avg + horiz_detail + vert_detail + diag_detail
                reconstructed[i, j+1]   = avg - horiz_detail + vert_detail - diag_detail
                reconstructed[i+1, j]   = avg + horiz_detail - vert_detail - diag_detail
                reconstructed[i+1, j+1] = avg - horiz_detail - vert_detail + diag_detail

        return reconstructed

# Example usage
if __name__ == "__main__":

    data = np.array([
        [52, 55, 61, 66, 70, 61, 64, 73],
        [63, 59, 66, 90, 109, 85, 69, 72],
        [62, 59, 68, 113, 144, 104, 66, 73],
        [63, 58, 71, 122, 154, 106, 70, 69],
        [67, 61, 68, 104, 126, 88, 68, 70],
        [79, 65, 60, 70, 77, 68, 58, 75],
        [85, 71, 64, 59, 55, 61, 65, 83],
        [87, 79, 69, 68, 65, 76, 78, 94]
    ])

    dwt_processor = ManualDWT(data)
    coeffs = dwt_processor.apply_dwt()
    print("DWT Coefficients:\n", coeffs)

    reconstructed_data = dwt_processor.apply_idwt()
    print("Reconstructed Data (after IDWT):\n", np.round(reconstructed_data))
