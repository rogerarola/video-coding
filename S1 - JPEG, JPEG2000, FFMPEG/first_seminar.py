#nere g
class RGBtoYUV():
    def _init_(self, R, G, B):
        self.R = R
        self.G = G
        self.B = B
    
    def conversor(self):
        Y = (( 66 * self.R + 129 * self.G +  25 * self.B + 128) / 256) +  16
        U = ((-38 * self.R -  74 * self.G + 112 * self.B + 128) / 256) + 128
        V = ((112 * self.R -  94 * self.G -  18 * self.B + 128) / 256) + 128
        yuv=[Y,U,V]
        return yuv

rgb1=RGBtoYUV(16,128,128)
yuv_values=rgb1.conversor()
print("YUV values:", yuv_values)

class YUVtoRGB():
    def _init_(self, Y, U, V):
        self.Y = Y - 16
        self.U = U - 128
        self.V = V - 128
    
    def conversor(self):
        R = 1.164 * self.Y + 1.596 * self.V
        G = 1.164 * self.Y - 0.392 * self.U - 0.813 * self.V
        B = 1.164 * self.Y + 2.017 * self.U
        rgb=[R,G,B]
        return rgb

rgb1=YUVtoRGB(97.625, 145.125, 79.5)
rgb_values=rgb1.conversor()
print("RGB values:", rgb_values)

import cv2
img = cv2.imread('../../GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/portrait-eyes.jpg')
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# resize
import ffmpeg
ffmpeg.input(img).filter('scale', 320, 240).output('../../GitHub/video-coding/S1 - JPEG, JPEG2000, FFMPEG/portrait-eyes-resized.jpg').run()