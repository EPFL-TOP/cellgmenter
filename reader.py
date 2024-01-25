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
def nd2reader_getSampleMetadata(path):
    stack = nd2.reader.ND2Reader(path)
    metadata = stack.metadata
    metadata_dict={}
    metadata_dict['number_of_frames']       = metadata['num_frames']
    metadata_dict['number_of_channels']     = len(metadata['channels'])
    metadata_dict['experiment_description'] = metadata['experiment']['description']
    metadata_dict['name_of_channels']       = ''
    metadata_dict['date']                   = metadata['date']
    
    for i in range(len(metadata['channels'])):
        if i<len(metadata['channels'])-1: metadata_dict['name_of_channels'] += metadata['channels'][i]+', '
        else: metadata_dict['name_of_channels'] += metadata['channels'][i]
    
    print('stack.metadata = ',stack.metadata)
    return metadata_dict

#_______________________________________________
def nd2reader_getFrameMetadata(path):
    stack = nd2.reader.ND2Reader(path)
    metadata = stack.metadata
    timesteps = stack.timesteps.tolist()
    metadata_dict={}
    metadata_dict['time'] = timesteps
    metadata_dict['x_pos'] = metadata['x_coordinates']
    metadata_dict['y_pos'] = metadata['y_coordinates']
    metadata_dict['z_pos'] = metadata['z_coordinates']
    metadata_dict['height'] = metadata['height']
    metadata_dict['width'] = metadata['width']

    return metadata_dict


#_______________________________________________
def nd2reader_getFrames(path):
    stack = nd2.reader.ND2Reader(path)
    metadata = stack.metadata
    sizes = stack.sizes
    if 't' not in sizes: sizes['t'] = 1
    if 'z' not in sizes: sizes['z'] = 1
    if 'c' not in sizes: sizes['c'] = 1
    stack.bundle_axes = 'zcyx'
    stack.iter_axes = 't'
    n = len(stack)

    shape = (sizes['t'], sizes['z'], sizes['c'], sizes['y'], sizes['x'])
    image  = np.zeros(shape, dtype=np.float32)

    for i in range(n):
        image[i] = stack.get_frame(i)
    image = np.squeeze(image)
    print('image shape in nd2raeader: ',image.shape)
    stack.close()
    return image, metadata['channels']

#_______________________________________________
def nd2reader(path):
    #https://github.com/cwood1967/napari-nikon-nd2/blob/main/napari_nikon_nd2/_reader.py
    stack = nd2.reader.ND2Reader(path)
    sizes = stack.sizes
    print ('sizes=',sizes)
    print ('stack=',stack)
    print('stack type=',type(stack))
    print('events=',stack.events)
    print('f rate---------------',stack.frame_rate)
    print('f timesteps---------------',stack.timesteps/60000.)
    print('len stack',len(stack))
    print('stack.metadata = ',stack.metadata)

    #parser=nd2.parser.Parser(path)
    #print('parser=',parser)
    #metadata=nd2.raw_metadata.RawMetadata(path)
    #print(metadata)
    #sys.exit(3)
    #print('--',len(stack))
    if 't' not in sizes: sizes['t'] = 1
    if 'z' not in sizes: sizes['z'] = 1
    if 'c' not in sizes: sizes['c'] = 1
    print (sizes)    

    stack.bundle_axes = 'zcyx'
    stack.iter_axes = 't'
    #print(stack.metadata)
    n = len(stack)

    shape = (sizes['t'], sizes['z'], sizes['c'], sizes['y'], sizes['x'])
    image  = np.zeros(shape, dtype=np.float32)

    for i in range(n):
        image[i] = stack.get_frame(i)
        #print('-------------- i=',i,'    ',stack.get_frame(i).metadata)
    image = np.squeeze(image)
    print(image.shape)
    stack.close()
    return image


if __name__ == "__main__":
    import sys
    nd2reader(sys.argv[1])
