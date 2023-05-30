import glob
import os, sys
import reader as myread
import numpy as np
import json
import matplotlib.pyplot as plt
from skimage import io

path='/Users/helsens/data/singleCell/metadata/*'
outtraining='/Users/helsens/data/singleCell/training/'


#---------------------------------------
def main():
    poslist=get_poslist()
    for pos in poslist:
        print(pos)
        maketraining(pos)


#---------------------------------------
def crop(image, x1, x2, y1, y2):
    if x2 == -1:
        x2=image.shape[1]-1
    if y2 == -1:
        y2=image.shape[0]-1

    mask = np.zeros(image.shape)
    mask[y1:y2+1, x1:x2+1]=1
    m = mask>0

    return image[m].reshape((y2+1-y1, x2+1-x1))

#---------------------------------------
def get_poslist():
    poslist=[]
    projects=glob.glob(path)
    for pr in projects:
        print('project ',pr)
        positions=glob.glob(os.path.join(path,pr,'*','position_data.json'))
        positions.sort()
        for pos in positions:
            poslist.append(pos)
    return poslist

#---------------------------------------
def maketraining(pos):
    imagepath=pos.replace('/metadata','').replace('/position_data.json','')
    imagepath+='.nd2'
    image = myread.nd2reader(imagepath)
    box=40
    f = open(pos)
    data = json.load(f)

    for cell in data:

        for label in ['maxima', 'minima', 'rising', 'falling']:

            for ind in data[cell][label]:
                center=data[cell]['center'][ind]
                image_cropped = crop(image[ind][0], int(center[1])-box, int(center[1])+box, int(center[0])-box, int(center[0])+box)
                outname=os.path.join(outtraining, label, '{}_tf{}_{}.jpeg'.format(imagepath.split('/')[-1].replace('.nd2',''),ind, cell))
                plt.imsave(outname, image_cropped, cmap='gray')
                #io.imsave(outname, image_cropped)
                #sys.exit()

        osc_end=data[cell]['oscilations_end']
        if osc_end<0:continue
        for ind in range(len(data[cell]['center'])):
            
            center=data[cell]['center'][ind]
            image_cropped = crop(image[ind][0], int(center[1])-box, int(center[1])+box, int(center[0])-box, int(center[0])+box)
            label='osc'
            if ind>osc_end: label='no_osc'

            outname=os.path.join(outtraining, label, '{}_tf{}_{}.jpeg'.format(imagepath.split('/')[-1].replace('.nd2',''),ind, cell))
            plt.imsave(outname, image_cropped, cmap='gray')


if __name__ == "__main__":
    main()

