import nd2reader as nd2
import numpy as np
from skimage.io import imread
#import nd2

#_______________________________________________
def tifreader(path):
    image = imread(path)
    image = image.transpose(0,3,1,2)
    return image


#def nd2reader_test(path):
#    image = nd2.imread(path)
#    print(image.shape)
#    return image

#_______________________________________________
def nd2reader(path):
    #https://github.com/cwood1967/napari-nikon-nd2/blob/main/napari_nikon_nd2/_reader.py
    stack = nd2.ND2Reader(path)
    sizes = stack.sizes
    print (sizes)
    print('--',len(stack))
    if 't' not in sizes: sizes['t'] = 1
    if 'z' not in sizes: sizes['z'] = 1
    if 'c' not in sizes: sizes['c'] = 1
    print (sizes)    

    stack.bundle_axes = 'zcyx'
    stack.iter_axes = 't'
    print(stack.metadata)
    n = len(stack)

    shape = (sizes['t'], sizes['z'], sizes['c'], sizes['y'], sizes['x'])
    image  = np.zeros(shape, dtype=np.float32)

    for i in range(n):
        image[i] = stack.get_frame(i)
        #print(stack.get_frame(i).metadata)
    image = np.squeeze(image)
    return image


if __name__ == "__main__":
    import sys
    nd2reader(sys.argv[1])
