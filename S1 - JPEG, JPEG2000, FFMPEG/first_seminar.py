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
image_path = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/landscape.jpg'  # Replace with the path to your JPEG image
serpentine(image_path)



