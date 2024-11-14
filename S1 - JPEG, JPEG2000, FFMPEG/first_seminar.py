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
    subprocess.run(command,check=True) 

#input_image = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/image.jpg'
#output_image = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/output_320x240.jpg'
#resize_image(input_image, output_image, 320, 240)


import subprocess
import numpy as np
from PIL import Image

im_landscape = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/landscape.jpg'

def serpentine(image_path):
    
    output_8x8 = 'C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/output_8x8.jpg'
    resize_image(im_landscape, output_8x8, 8, 8)
    img = Image.open(output_8x8)
    img_gray = img.convert("L")
    gray_image_path = "C:/Users/Nerea/OneDrive/Escritorio/Uni/4t/CAV/video-coding/S1 - JPEG, JPEG2000, FFMPEG/output_8x8_gray.jpg"
    img_gray.save(gray_image_path)
    img_matrix = np.array(img_gray)
    print(img_matrix)
    output = zig_zag_matrix(img_matrix)
    return output

def zig_zag_matrix(mat):
    n = len(mat)
    m = len(mat[0])
    row = 0
    col = 0
    result = []

    # Boolean variable that is true if we need
    # to increment 'row' value;
    # otherwise, false if we increment 'col' value.
    row_inc = False

    # Process the first half of the zig-zag pattern
    mn = min(m, n)
    for length in range(1, mn + 1):
        for i in range(length):
            result.append(mat[row][col])

            if i + 1 == length:
                break

            # If row_inc is true, increment row 
            # and decrement col;
            # otherwise, decrement row and increment col.
            if row_inc:
                row += 1
                col -= 1
            else:
                row -= 1
                col += 1

        if length == mn:
            break

        # Update row or col value based on the
        # last increment
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

    # Process the second half of the zig-zag pattern
    MAX = max(m, n) - 1
    for diag in range(MAX, 0, -1):
        length = mn if diag > mn else diag
        for i in range(length):
            result.append(mat[row][col])

            if i + 1 == length:
                break

            # Update row or col value based on the last increment
            if row_inc:
                row += 1
                col -= 1
            else:
                col += 1
                row -= 1

        # Update row and col based on position in the matrix
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

    return result

array=serpentine(im_landscape)
print(array)





