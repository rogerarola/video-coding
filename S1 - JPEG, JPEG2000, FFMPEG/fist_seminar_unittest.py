import unittest
import numpy as np
from first_seminar import RGBtoYUV, YUVtoRGB, resize_image, serpentine, zig_zag_matrix, printRLE, DCT, DWT

class TestRGBtoYUV(unittest.TestCase):
    def test_rgb_to_yuv_conversion(self):
        rgb1 = RGBtoYUV(16, 128, 128)
        yuv_values = rgb1.conversor()
        expected_yuv = [88.5, 143.625, 104.875]
        self.assertAlmostEqual(yuv_values[0], expected_yuv[0], places=1)
        self.assertAlmostEqual(yuv_values[1], expected_yuv[1], places=1)
        self.assertAlmostEqual(yuv_values[2], expected_yuv[2], places=1)

class TestYUVtoRGB(unittest.TestCase):
    def test_yuv_to_rgb_conversion(self):
        yuv1 = YUVtoRGB(97.625, 145.125, 79.5)
        rgb_values = yuv1.conversor()
        expected_rgb = [180.234, 127.953, 97.877]
        self.assertAlmostEqual(rgb_values[0], expected_rgb[0], places=1)
        self.assertAlmostEqual(rgb_values[1], expected_rgb[1], places=1)
        self.assertAlmostEqual(rgb_values[2], expected_rgb[2], places=1)

class TestImageResize(unittest.TestCase):
    def test_resize_image(self):
        input_path = "test_image.jpg"
        output_path = "output_test.jpg"
        resize_image(input_path, output_path, 320, 240)
        # Additional validation for output file dimensions would require checking file properties.

class TestSerpentine(unittest.TestCase):
    def test_zig_zag_matrix(self):
        matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        zigzag_result = zig_zag_matrix(matrix)
        expected = [1, 2, 4, 7, 5, 3, 6, 8, 9]
        self.assertEqual(zigzag_result, expected)

class TestRLE(unittest.TestCase):
    def test_printRLE(self):
        with self.assertLogs() as captured:
            printRLE("wwwwaaadexxxxxxywww")
            self.assertIn("w4a3d1e1x6y1w3", captured.output[0])

class TestDCT(unittest.TestCase):
    def test_dct_and_idct(self):
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
        reconstructed = dct_processor.idct()
        np.testing.assert_almost_equal(reconstructed, data, decimal=1)

class TestDWT(unittest.TestCase):
    def test_dwt_and_idwt(self):
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
        dwt_processor = DWT(data)
        coeffs = dwt_processor.dwt()
        reconstructed_data = dwt_processor.idwt()
        np.testing.assert_almost_equal(reconstructed_data, data, decimal=1)

if __name__ == "__main__":
    unittest.main()
