class RGBtoYUV:
    def __init__(self, R, G, B):
        self.R = R
        self.G = G
        self.B = B

    def conversor(self):
        Y = ((66 * self.R + 129 * self.G + 25 * self.B + 128) / 256) + 16
        U = ((-38 * self.R - 74 * self.G + 112 * self.B + 128) / 256) + 128
        V = ((112 * self.R - 94 * self.G - 18 * self.B + 128) / 256) + 128
        return [Y, U, V]


class YUVtoRGB:
    def __init__(self, Y, U, V):
        self.Y = Y - 16
        self.U = U - 128
        self.V = V - 128

    def conversor(self):
        R = 1.164 * self.Y + 1.596 * self.V
        G = 1.164 * self.Y - 0.392 * self.U - 0.813 * self.V
        B = 1.164 * self.Y + 2.017 * self.U
        return [R, G, B]
    