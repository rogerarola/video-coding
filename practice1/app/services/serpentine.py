import numpy as np
from PIL import Image
from .resize_image import resize_image

def serpentine(image_path):
    output_8x8 = "/tmp/output_8x8.jpg"
    gray_image_path = "/tmp/output_8x8_gray.jpg"
    
    # Ensure FFmpeg overwrites existing files
    resize_image(image_path, output_8x8, 8, 8)
    img = Image.open(output_8x8)
    img_gray = img.convert("L")
    img_gray.save(gray_image_path)
    img_matrix = np.array(img_gray)

    if img_matrix.shape[0] != 8 or img_matrix.shape[1] != 8:
        raise ValueError("Image must be resized to an 8x8 matrix for zig-zag traversal.")
    
    zigzag_result = zig_zag_matrix(img_matrix)
    # Convert numpy types to Python-native types
    zigzag_result = [int(x) for x in zigzag_result]
    return zigzag_result


def zig_zag_matrix(mat):
    n, m = mat.shape
    result = []
    row, col = 0, 0
    direction_up = True

    while len(result) < n * m:
        result.append(mat[row, col])
        if direction_up:
            if col == m - 1:
                row += 1
                direction_up = False
            elif row == 0:
                col += 1
                direction_up = False
            else:
                row -= 1
                col += 1
        else:
            if row == n - 1:
                col += 1
                direction_up = True
            elif col == 0:
                row += 1
                direction_up = True
            else:
                row += 1
                col -= 1

    return result
