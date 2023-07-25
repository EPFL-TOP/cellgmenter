from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing, binary_dilation
from scipy.ndimage import binary_fill_holes
import numpy as np
import numba as nb
from numba.types import bool_
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt

import json
import glob, os, sys, math
from pathlib import Path

gpu = False
sigma = 50
minimum_size=250
# Functions
def normalize_background(img, sigma, gpu):
    
    """
    This function loads an image and performs the division of the input by a blurred filtered version of itself.
    """
    intensity_normalized = None
    
    if gpu:
        pushed = cle.push(img)
        intensity_normalized = cle.divide_by_gaussian_background(pushed, result_substract, sigma, sigma, 0)
        intensity_normalized = np.asarray(intensity_normalized) #img is pulled from GPU memory
    
    else:
        intensity_normalized = cle.divide_by_gaussian_background(img, intensity_normalized, sigma,sigma,0)
        intensity_normalized = np.asarray(intensity_normalized)
    
    return intensity_normalized

#_______________________________________________
def apocSeg(clf, input, outdir, npix=300):
    image=input[:,0,:,:]
    normalize = normalize_background(image, sigma, gpu)
    for count in range(normalize.shape[0]):
        prediction = np.asarray(clf.predict(normalize[count])-1,dtype=np.uint8)
        dilated = binary_dilation(prediction, disk(2))
        closed = binary_closing(dilated, disk(4))
        filled = binary_fill_holes(closed).astype(int)
        label_im = label(filled)

        regions=regionprops(label_im)
        cells=[]
        for r in regions:
            if r.area>npix:
                cells.append(r)
        
        previous_labels=[]
        if count>0:
            previous_labels = glob.glob(os.path.join(outdir, 'mask_tf{}_apoc_cell*.json'.format(count-1)))
        extracount=0
        for c in range(len(cells)):
            minDist=1000000000
            celllabel='cell{}'.format(c)
            for pl in previous_labels:
                pl_file = open(pl)
                pl_data = json.load(pl_file)
                    
                dist=math.sqrt((pl_data['center'][0]-cells[c].centroid[0])*(pl_data['center'][0]-cells[c].centroid[0])+
                        (pl_data['center'][1]-cells[c].centroid[1])*(pl_data['center'][1]-cells[c].centroid[1]))
                if dist<minDist: 
                    minDist=dist
                    celllabel=pl_data['label']

            if minDist>50 and minDist<100000000:
                celllabel='cell{}'.format(len(previous_labels)+extracount)
                extracount+=1
            intensities=[]
            intensities_list=[]
            for i in range(len(input[count])):
                intensity=0
                intensity_list=[]
                for coord in cells[c].coords:
                    intensity+=input[count][i][coord[0]][coord[1]]
                    intensity_list.append(input[count][i][coord[0]][coord[1]])
                intensities.append(intensity)
                intensities_list.append(intensity_list)

            mask0=np.zeros(image[0].shape, dtype=bool)
            for coord in cells[c].coords:
                mask0[coord[0]][coord[1]]=True
            cs=plt.contour(mask0, [0.5],linewidths=1.2,  colors='red')
            contcoords= cs.allsegs[0][0]

            dic={
                'npixels':cells[c].area,
                'center':cells[c].centroid,
                'nchannels':len(input[count]),
                'intensity':intensities,
                'label':celllabel,
                'coords':cells[c].coords,
                'xcoords':contcoords[:,0],
                'ycoords':contcoords[:,1],
                'intensity_list':intensities_list


            }
            json_object = json.dumps(dic, cls=NpEncoder)

            # Writing to <out>.json
            outname=os.path.join(outdir, 'mask_tf{}_apoc_{}.json'.format(count,celllabel))
            with open(outname, "w") as outfile:
                outfile.write(json_object)

#_______________________________________________
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)




#_______________________________________________
@nb.njit(fastmath = True)
def fastiter(image, thr=2., delta=1):
    img_seeds=np.zeros(image.shape, dtype=bool_)
    for i in range(image.shape[0]):
        bkg=[]
        for ii in range(-5,5):
            iii=ii+i
            if iii<0 or iii>image.shape[0]-1:continue
            for jj in range(0,25):
                bkg.append(image[iii][jj])
        bkg=np.array(bkg)
        std=np.std(bkg)
        for j in range(image.shape[1]):
            sig=[]
            for id in range(-delta, delta+1):
                if id+i<0 or id+i>image.shape[0]-1:continue
                for jd in range(-delta, delta+1):
                    if jd+j<0 or jd+j>image.shape[1]-1:continue
                    sig.append(image[i+id][j+jd])

            if np.std(np.array(sig))>thr*std:
                img_seeds[i][j]=True
    return img_seeds





#_______________________________________________
def simpleSeg(img,  outdir, count, thr=2., delta=1, npix=400):
    image=img[0]
    img_seeds=fastiter(image, thr, delta)

    #dilated = binary_dilation(img_seeds, disk(2))
    closed = binary_closing(img_seeds, disk(4))
    filled = binary_fill_holes(closed).astype(int)
    label_im = label(filled)

    regions=regionprops(label_im)
    cells=[]
    for r in regions:
        if r.area>npix:
            cells.append(r)
    
    previous_labels=[]
    if count>0:
        previous_labels = glob.glob(os.path.join(outdir, 'mask_tf{}_thr{}delta{}_cell*.json'.format(count-1,thr,delta)))
    extracount=0
    for c in range(len(cells)):
        minDist=1000000000
        celllabel='cell{}'.format(c)
        for pl in previous_labels:
            pl_file = open(pl)
            pl_data = json.load(pl_file)
                
            dist=math.sqrt((pl_data['center'][0]-cells[c].centroid[0])*(pl_data['center'][0]-cells[c].centroid[0])+
                    (pl_data['center'][1]-cells[c].centroid[1])*(pl_data['center'][1]-cells[c].centroid[1]))
            if dist<minDist: 
                minDist=dist
                celllabel=pl_data['label']

        if minDist>50 and minDist<100000000:
            celllabel='cell{}'.format(len(previous_labels)+extracount)
            extracount+=1
        intensities=[]
        intensities_list=[]

        for i in range(len(img)):
            intensity=0
            intensity_list=[]
            for coord in cells[c].coords:
                intensity+=img[i][coord[0]][coord[1]]
                intensity_list.append(input[count][i][coord[0]][coord[1]])
            intensities.append(intensity)
            intensities_list.append(intensity_list)

        mask0=np.zeros(img[0].shape, dtype=bool)
        for coord in cells[c].coords:
            mask0[coord[0]][coord[1]]=True
        cs=plt.contour(mask0, [0.5],linewidths=1.2,  colors='red')
        contcoords= cs.allsegs[0][0]
        
        dic={
            'npixels':cells[c].area,
            'center':cells[c].centroid,
            'nchannels':len(img),
            'intensity':intensities,
            'label':celllabel,
            'coords':cells[c].coords,
            'xcoords':contcoords[:,0],
            'ycoords':contcoords[:,1],
            'intensity_list':intensities_list
        }
        json_object = json.dumps(dic, cls=NpEncoder)

        # Writing to <out>.json
        outname=os.path.join(outdir, 'mask_tf{}_thr{}delta{}_{}.json'.format(count,thr,delta,celllabel))
        with open(outname, "w") as outfile:
            outfile.write(json_object)


    #Take simple seg as default
    celldic={}
    celllist=glob.glob(os.path.join(outdir , 'mask_tf{}_thr{}delta{}_cell*.json'.format(count,2.,2)))
    celllist.sort()
    for cellid, cell in enumerate(celllist):
 
        celldic[os.path.split(cell)[-1].split('_')[-1].replace('.json','')]={
			'mask':cell,
        	'valid':True, #Set to False if user find out this cell is bad
		    'alive':True, #True/False
			'status':'single',  #'single, doublenuclei, multiplecells, pair from a menu'
        	'isdividing':False, #True/False', can span over multiple TF
            }
                

    timeframedic={
        'skipframe':False, #false by default
        'cells':celldic
            }
    jsontf_object = json.dumps(timeframedic, indent=4)
    outnametf=os.path.join(outdir, 'metadata_tf{}.json'.format(count))
    if not os.path.isfile(outnametf):
        with open(outnametf, "w") as outfiletf:
            outfiletf.write(jsontf_object)