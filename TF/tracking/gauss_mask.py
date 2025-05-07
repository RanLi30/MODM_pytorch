import numpy as np
import matplotlib.pyplot as plt

def gauss_mask(shape,center,sigma):
    center_x = center[0]
    center_y = center[1]

    mask_x = np.tile(center_x, [shape[1],shape[0]])
    mask_y = np.tile(center_y, [shape[1],shape[0]])

    x1 = np.arange(shape[0])
    x_map = np.tile(x1, [shape[1],1])

    y1 = np.arange(shape[1])
    y_map = np.tile(y1, [shape[0], 1])
    y_map = np.transpose(y_map)

    Gmask = np.sqrt((x_map-mask_x)**2+(y_map-mask_y)**2)
    Gmask = np.exp(-0.5*Gmask/sigma)

    return Gmask

if __name__=='__main__':
    shape = [960,660]
    center = [500,300]
    sigma = 30

    gausmask = gauss_mask(shape,center,sigma)

    plt.figure()
    plt.imshow(gausmask,plt.cm.gray)
    plt.show()