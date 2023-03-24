from skimage.measure import label, regionprops
from skimage.morphology import disk, binary_closing
from scipy.ndimage import binary_fill_holes
import numpy as np
import json
import glob, os, sys, math


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
def simpleSeg(img,  outdir, count, thr=3.5, delta=1, npix=100):
    print('========== segmentation')
    if count>17:return
    image=img[0]
    img_seeds=np.zeros(image.shape, dtype=bool)
    for i in range(len(image)):
        tmp=np.concatenate((image[i-5:i+5, 0:25].flatten(),image[i-5:i+5, len(image)-25:len(image)].flatten()), axis=None)
        mean=np.mean(tmp)
        std=np.std(tmp)
        for j in range(len(image)):
            tmp2=image[i-delta:i+delta, j-delta:j+delta]

            flat=tmp2.flatten()
            if np.std(flat)>thr*std:
                img_seeds[i][j]=True

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
        previous_labels = glob.glob(outdir+'/mask_tf{}_thr{}delta{}_cell*.json'.format(count-1,thr,delta))
    extracount=0
    for c in range(len(cells)):
        minDist=1000000000
        celllabel='cell{}'.format(c)
        for pl in previous_labels:
            pl_file = open(pl)
            pl_data = json.load(pl_file)
                
            dist=math.sqrt((pl_data['center'][0]-cells[c].centroid[0])*(pl_data['center'][0]-cells[c].centroid[0])+
                    (pl_data['center'][1]-cells[c].centroid[1])*(pl_data['center'][1]-cells[c].centroid[1]))
            print('=================  ',dist)
            if dist<minDist: 
                minDist=dist
                celllabel=pl_data['label']

        if minDist>50 and minDist<100000000:
            celllabel='cell{}'.format(len(previous_labels)+extracount)
            extracount+=1
        intensities=[]
        for i in range(len(img)):
            intensity=0
            for coord in cells[c].coords:
                intensity+=img[i][coord[0]][coord[1]]
            intensities.append(intensity)
        dic={
            'npixels':cells[c].area,
            'center':cells[c].centroid,
            'nchannels':len(img),
            'intensity':intensities,
            'label':celllabel,
            'coords':cells[c].coords

        }
        json_object = json.dumps(dic, cls=NpEncoder)

        # Writing to <out>.json
        outname=outdir+'/mask_tf{}_thr{}delta{}_{}.json'.format(count,thr,delta,celllabel)
        with open(outname, "w") as outfile:
            outfile.write(json_object)


    celldic={}
    celllist=glob.glob(outdir+'/mask_tf{}_thr{}delta{}_cell*.json'.format(count,2.,2))
    celllist.sort()
    for cellid, cell in enumerate(celllist):
 
        celldic['cell{}'.format(cellid)]={
			'mask':cell,
        	'valid':'True', #Set to False if user find out this cell is bad
		    'alive':'True', #True/False
			'status':'NA',  #'single, doublenuclei, multiplecells, pair from a menu'
        	'isdividing':'False', #True/False', can span over multiple TF
            }
                

    timeframedic={
        'skipframe':'False', #false by default
        'cells':celldic
            }
    jsontf_object = json.dumps(timeframedic, indent=4)
    outnametf=outdir+'/metadata_tf{}.json'.format(count)
    if not os.path.isfile(outnametf):
        with open(outnametf, "w") as outfiletf:
            outfiletf.write(jsontf_object)