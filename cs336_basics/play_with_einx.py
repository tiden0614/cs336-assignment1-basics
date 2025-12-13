import numpy as np
import einx

if __name__ == '__main__':
    x = np.ones((256, 256, 3), dtype="uint8")
    a = einx.rearrange("(s p)... c -> s... p... c", x, p=8)
    print(a.shape)