"""
Loading Nikon D3400 image from data/campus.nef ...
Scaling with darkness 150, saturation 4095, and
multipliers 2.394531 1.000000 1.597656 1.000000
Building histograms...
Writing data to data/campus.tiff ...

(4016, 6016)

31,454,184
5,459,148

5.7617
unit16 - 16 bit integer - 0 - 65535

dcraw -a -r 2 1 1.5 1 -o 1 -q 1 -f data/campus.nef && magick data/campus.ppm data/campus.png

"""


import numpy as np
from scipy.interpolate import interp2d
import skimage
from skimage.color import rgb2gray




# Manual White Balance
def white_balance_patch(data, patch_x, patch_y):
    x1, x2 = patch_x
    y1, y2 = patch_y

    patch = data[y1:y2, x1:x2]
    R, G, B = patch[:,:,0], patch[:,:,1], patch[:,:,2]

    print("KK", patch.shape)
    r_val = R[R != 0].mean()
    g_val = G[G != 0].mean()
    b_val = B[B != 0].mean()

    
    print(r_val, g_val, b_val)
    R = data[:,:,0]/r_val
    G = data[:,:,1]/g_val
    B = data[:,:,2]/b_val

    return (R + G + B)

# White balance with grey world, white world and rgb scale
def white_balance(data, type):
    """
    0 -> Grey
    1 -> White
    2 -> Custom RGB
    """

    R, G, B = data[:,:,0], data[:,:,1], data[:,:,2]


    if type == 0:
        r_val = R[R != 0].mean()
        g_val = G[G != 0].mean()
        b_val = B[B != 0].mean()
    if type == 1:
        r_val = R.max()
        g_val = G.max()
        b_val = B.max()
    if type == 2:
        R = R * r_scale
        B = B * b_scale
        G = G * g_scale
    else:    
        R = R*(g_val/r_val)
        B = B*(g_val/b_val)

    return (R + G + B)


# Apply bayer pattern
def bayer_pattern(image):
    
    masks = dict((color, np.zeros(image.shape))for color in ['Red', 'Green', 'Blue'])
    masks['Red'][0::2, 0::2] = 1
    masks['Green'][0::2, 1::2] = 1
    masks['Green'][1::2, 0::2] = 1
    masks['Blue'][1::2, 1::2] = 1
    
    
    for color in ['Red', 'Green', 'Blue']:
        masks[color] = masks[color].astype(bool)

    R = image * masks['Red']
    G = image * masks['Green']
    B = image * masks['Blue']

    return [np.stack([R, G, B],axis=2), masks]

# Apply Demosiacing
def demosiac(image):
    
    height, width = image.shape
    
    x = np.arange(0, width, 2)  
    y = np.arange(0, height, 2)
    X,Y = np.meshgrid(x,y)
    R_interp = interp2d(x, y, image[Y,X], kind='linear')

    x = np.arange(1, width, 2)  
    y = np.arange(1, height, 2)
    X,Y = np.meshgrid(x,y)
    B_interp = interp2d(x, y, image[Y,X], kind='linear')

    x = np.arange(1, width, 2)  
    y = np.arange(0, height, 2)
    X,Y = np.meshgrid(x,y)
    G_interp_1 = interp2d(x, y, image[Y,X], kind='linear')

    x = np.arange(0, width, 2)  
    y = np.arange(1, height, 2)
    X,Y = np.meshgrid(x,y)
    G_interp_2 = interp2d(x, y, image[Y,X], kind='linear')
    
    x = np.arange(0, width)  
    y = np.arange(0, height)

    R = R_interp(x, y)
    G = (G_interp_1(x, y)  +  G_interp_2(x, y))/2
    B = B_interp(x, y)
    
    return np.dstack([R, G, B])


black_thresh = 150
window_thresh = 4095


g_scale = 1.000000
r_scale = 2.394531
b_scale = 1.597656
# 0 = Grey, 1 = White, 2 = RGB
type = 0
x_coords = [3375, 3391]
y_coords = [3033,3177]


# Python initials
data = skimage.io.imread('data/campus.tiff')

data = data.astype('double')
height, width = data.shape
 
 # Linearization
data = (data - black_thresh)/(window_thresh - black_thresh)
data[data < 0] = 0
data[data > 1] = 1

# Apply bayer pattern
out, masks = bayer_pattern(data)

# Apply White Balance
out = white_balance_patch(out, patch_x = x_coords, patch_y = y_coords)
out = np.clip(out,0.,1.)

skimage.io.imsave('white_bal_grey.png', (out*255).astype('uint8'))

# Demosaic
out = demosiac(out)
out = np.clip(out,0.,1.)

skimage.io.imsave('demos.png', (out*255).astype('uint8'))


grey = rgb2gray(out)
mean_grey = np.mean(grey)
scaling_factor = 0.25/mean_grey

output = out*scaling_factor
output = np.clip(output,0.,1.)
skimage.io.imsave('scaled_25.png', (output*255).astype('uint8'))


# Color Correction
def color_correction(img, ccm):
    return np.dot(img, ccm.T)

M_xyz_cam = np.array([[6988,-1384,-714],[-5631,13410,2447],[-1485,2204,7318]]) /10000.0

M_rgb_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
[0.2126729, 0.7151522, 0.0721750],
[0.0193339, 0.1191920, 0.9503041]])

M_srgb_xyz = np.matmul(M_xyz_cam, M_rgb_xyz)

M_srgb_xyz = M_srgb_xyz / M_srgb_xyz.sum(axis=1, keepdims=True)

M_srgb_xyz_inv = np.linalg.inv(M_srgb_xyz)

linear_out = color_correction(out, M_srgb_xyz_inv)

linear_out = np.clip(linear_out,0.,1.)
skimage.io.imsave('color_correction.png', (linear_out*255).astype('uint8'))


# Gamma reduction
type1 = linear_out <= 0.0031308
type2 = linear_out > 0.0031308

linear_out[type1] = 12.92 * linear_out[type1]
linear_out[type2] = (1 + 0.055)*(linear_out[type2])**(1/2.4)  - 0.055

non_linear_out = linear_out

non_linear_out = np.clip(non_linear_out,0.,1.)
skimage.io.imsave('processed_image.png', (non_linear_out*255).astype('uint8'))
skimage.io.imsave('processed_image.jpeg', (non_linear_out*255).astype('uint8'), quality = 20)