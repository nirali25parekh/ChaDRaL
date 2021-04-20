import cv2
import numpy as np

# given an image, convert to R, G, B arrays
def split_into_rgb_channels(image):
    # BGR values
    #// image.shape (390, 600, 3) => width * height * channels
    red = image[:,:,2] # each pixel, index2
    green = image[:,:,1] # each pixel, index1
    blue = image[:,:,0] # each pixel, index0
    # red array takes out R values from each pixel (red = [[139 18 ... 37], [142 141 ..130]...[....]]) 
    # len(red) is width of image, len(red[0] is height of image)
    return red, green, blue

# image converted to three matrices of R, G, B colors
def decompose_matrix(file_path):
    image = cv2.imread(file_path)
    #// image.shape (390, 600, 3) => width * height * channels
    blue,green,red = split_into_rgb_channels(image)
    # now, blue, red, green are array of arrays. To access elements -> blue[0][0] => <class 'numpy.ndarray'>
    # blue, red, green are all 'm' length arrays each containing n elements

    # BGR is the order, hence 0->blue, 1->green, 2->red
    for values, channel in zip((red, green, blue), (2,1,0)):
        img = np.zeros((values.shape[0], values.shape[1]), dtype = np.uint8)
        img[:,:] = (values)
        if channel == 0:
            B = np.asmatrix(img)
        elif channel == 1:
            G = np.asmatrix(img)
        else:
            R = np.asmatrix(img)
    # now B, R, G are matrices -> to access elements, B[0,0]  => <class 'numpy.matrix'>
    return B,G,R