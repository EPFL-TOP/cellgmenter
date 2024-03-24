from skimage.measure import label, regionprops, find_contours
from skimage.morphology import disk, binary_closing, binary_dilation
from scipy.ndimage import binary_fill_holes
import numpy as np
import numba as nb
from numba.types import bool_
import pyclesperanto_prototype as cle
import matplotlib.pyplot as plt
import matplotlib
from pympler import asizeof

matplotlib.use('agg')

import json
import glob, os, sys, math
from pathlib import Path
import apoc
from scipy.ndimage import gaussian_filter
from skimage import filters as skfilters
from skimage import morphology
from skimage.measure import label, regionprops, find_contours

gpu = False
sigma = 50
minimum_size=250
version='0.0.1'

#_______________________________________________
def norm(img, tonorm=1):
    r = img*tonorm /img.mean()
    return r

#_______________________________________________
def get_ROIs_per_frame(image, thr=3.5):

    BF_image_filter_high = gaussian_filter(image,20)
    totry =np.abs(image-BF_image_filter_high)

    val = skfilters.threshold_otsu(totry)
    otsu = totry>val*thr

    closed = binary_closing(otsu, disk(2))
    filled = binary_fill_holes(closed).astype(int)

    res = morphology.white_tophat(filled, morphology.disk(4)) 

    filled = filled-res
    filled = morphology.dilation(filled, morphology.disk(15))

    label_im = label(filled)
    regions=regionprops(label_im)
    ROIs=[]
    for r in regions:
        if r.num_pixels>100 and r.num_pixels<15000:
            #Bounding box (min_row, min_col, max_row, max_col). 
            #Pixels belonging to the bounding box are in the half-open interval [min_row; max_row) and [min_col; max_col).
            ROIs.append(r.bbox)

    return ROIs

#_______________________________________________
def get_ROIs_per_sample(images):
    BF_images=images.transpose(1,0,2,3)
    BF_images=BF_images[0]
    for im in range(len(BF_images)):
        BF_images[im]=norm(BF_images[im])
    BF_images=BF_images.transpose(1,2,0)

    BF_image_std = np.std(BF_images, axis=2)
    edge_sobel = skfilters.sobel(BF_image_std)

    val = skfilters.threshold_otsu( gaussian_filter(edge_sobel, 2))
    otsu = edge_sobel>val
    closed = binary_closing(otsu, morphology.disk(4))
    filled = binary_fill_holes(closed).astype(int)

    res = morphology.white_tophat(filled, morphology.disk(8)) 

    filled = filled-res
    filled = morphology.dilation(filled, morphology.disk(10))

    label_im = label(filled)
    regions=regionprops(label_im)
    ROIs=[]
    for r in regions:
        if r.num_pixels>100 and r.num_pixels<15000:
            ROIs.append(r.bbox)

    return ROIs

#_______________________________________________
class customLocalThresholding_Segmentation:
    def __init__(self, threshold=2., delta=1, npix_min=400, npix_max=4000, min_row=-999, min_col=-999, max_row=-999, max_col=-999):
        self.threshold  = threshold
        self.delta      = delta
        self.npix_min   = npix_min
        self.npix_max   = npix_max
        self.algorithm_parameters = {'threshold':threshold, 'delta':delta, 'npix_min':npix_min, 'npix_max':npix_max}
        self.algorithm_type       = 'localthresholding'
        self.algorithm_version    = 'main'
        self.channels = None
        self.channel  = None
        self.min_row = min_row
        self.min_col = min_col
        self.max_row = max_row
        self.max_col = max_col        


    #_______________________________________________
    def get_param(self):
        return self.algorithm_parameters
    #_______________________________________________
    def get_type(self):
        return self.algorithm_type
    #_______________________________________________
    def get_version(self):
        return self.algorithm_version
    #_______________________________________________
    def set_channels(self, channel, channels):
        self.channel  = channel
        self.channels = channels
    
    #_______________________________________________
    def segmentation(self, img):
        if self.channel==None or self.channels==None:
            print("Can not segment, channel or channels is NoneType")
            return
        image=img[self.channel]
        #img_seeds=fastiter(image, self.delta, self.threshold)
        img_seeds=fastiter_range(image, self.delta, self.threshold, self.min_row, self.min_col, self.max_row, self.max_col)

        #dilated = binary_dilation(img_seeds, disk(2))
        closed   = binary_closing(img_seeds, disk(4))
        filled   = binary_fill_holes(closed).astype(int)
        label_im = label(filled)
        regions  = regionprops(label_im)

        contours=[]
        for r in regions:
        
            if r.num_pixels>self.npix_min and r.num_pixels<self.npix_max:
                contours.append(r)

        contour_list=build_contour_dict(contours, image, img, self.channels)

        for c in contour_list:
            c['algorithm_parameters'] = self.algorithm_parameters
            c['algorithm_type']       = self.algorithm_type
            c['algorithm_version']    = self.algorithm_version
        return contour_list

#_______________________________________________
def segmentation_test(img, thr, min_row, min_col, max_row, max_col):

    print(img)
    print('thr=',thr, '  min_row=',min_row, '  max_row=',max_row, '  min_col=',min_col, '  max_col=',max_col )
    
    BF_image_filter_high = gaussian_filter(img,20)
    totry = gaussian_filter(img,2)

    img_seeds=fastiter_range(totry, thr, min_row, min_col, max_row, max_col)

    #dilated = binary_dilation(img_seeds, disk(2))
    closed   = binary_closing(img_seeds, disk(4))
    filled   = binary_fill_holes(closed).astype(int)
    label_im = label(filled)
    regions  = regionprops(label_im)

    contour=None
    max_pix=0
    for r in regions:
    
        if r.area>max_pix:
            contour=r
            max_pix=r.area

    return contour


#_______________________________________________
@nb.njit(fastmath = True)
def fastiter_range(image, threshold, min_row, min_col, max_row, max_col):
    img_seeds=np.zeros(image.shape, dtype=bool_)
    for i in range(min_row, max_row+1):
        bkg=[]
        for ii in range(i-1,i+2):
            if ii<0 or ii>image.shape[0]-1:continue
            for jj in range(min_col-3,min_col):
                bkg.append(image[ii][jj])

            for jj in range(max_col+1,max_col+4):
                bkg.append(image[ii][jj])

        bkg=np.array(bkg)
        std=np.std(bkg)
        mean=np.mean(bkg)

        for j in range(min_col, max_col+1):
            print('i=',i,'  j=',j,'  int=',image[i][j],'  mean=',mean,'  std=',std)
            if image[i][j]>mean+threshold*std or image[i][j]<mean-threshold*std:
                img_seeds[i][j]=True
                print('sig')

            #sig=[]
            #for id in range(-delta, delta+1):
            #    if id+i<0 or id+i>image.shape[0]-1:continue
            #    for jd in range(-delta, delta+1):
            #        if jd+j<0 or jd+j>image.shape[1]-1:continue
            #        sig.append(image[i+id][j+jd])

            #if np.std(np.array(sig))>threshold*std:
            #    img_seeds[i][j]=True
    return img_seeds


#_______________________________________________
@nb.njit(fastmath = True)
def fastiter(image, delta, threshold):
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

            if np.std(np.array(sig))>threshold*std:
                img_seeds[i][j]=True
    return img_seeds


#_______________________________________________
@nb.njit(fastmath = True)
def fastiter_mean(image, delta, threshold_std):
    img_seeds=np.zeros(image.shape, dtype=bool_)
    img_mean=np.zeros(image.shape)
    img_std=np.zeros(image.shape)

    for i in range(image.shape[0]):
        bkg=[]
        for ii in range(-5,5):
            iii=ii+i
            if iii<0 or iii>image.shape[0]-1:continue
            for jj in range(0,25):
                bkg.append(image[iii][jj])
        bkg=np.array(bkg)
        std=np.std(bkg)
        mean=np.mean(bkg)
        for j in range(image.shape[1]):
            sig=[]
            for id in range(-delta, delta+1):
                if id+i<0 or id+i>image.shape[0]-1:continue
                for jd in range(-delta, delta+1):
                    if jd+j<0 or jd+j>image.shape[1]-1:continue
                    sig.append(image[i+id][j+jd])

            #if np.std(np.array(sig))>threshold_std*std and np.abs(np.mean(np.array(sig))-mean)/mean>1.2:
            if i>200 and i<300 and j>200 and j<300 and np.abs(np.mean(np.array(sig))-mean)>1.5:
                print(i,'  ',j,'  ',np.abs(np.mean(np.array(sig))-mean),'  ',np.std(np.array(sig)),'  ',std)
            if np.abs(np.mean(np.array(sig))-mean)>1.5 and np.std(np.array(sig))>threshold_std*std:
                img_seeds[i][j]=True
            img_mean[i][j]=np.mean(np.array(sig))
            img_std[i][j]=np.std(np.array(sig))
    return img_seeds,img_mean,img_std


#_______________________________________________
def build_contour_dict(contours, image, img, channels):
    out_contours=[]
    for c in range(len(contours)):
        single_pixels_inside={'x':[], 'y':[], 'z':[], 'intensity':{}}
        all_pixels_inside={'sum_intensity':{}, 'std_intensity':{}, 'mean_intensity':{}}
        single_pixels_contour={'x':[], 'y':[], 'z':[], 'intensity':{}}
        all_pixels_contour={'sum_intensity':{}, 'std_intensity':{}, 'mean_intensity':{}}

        mask0=np.zeros(image.shape, dtype=bool)

        for ch in channels:
            single_pixels_inside['intensity'][ch]=[]
            all_pixels_inside['sum_intensity'][ch]=0
            all_pixels_inside['std_intensity'][ch]=0
            all_pixels_inside['mean_intensity'][ch]=0

            single_pixels_contour['intensity'][ch]=[]
            all_pixels_contour['sum_intensity'][ch]=0
            all_pixels_contour['std_intensity'][ch]=0
            all_pixels_contour['mean_intensity'][ch]=0

        for p in contours[c].coords:
            mask0[p[0]][p[1]]=True
            single_pixels_inside['x'].append(int(p[0]))
            single_pixels_inside['y'].append(int(p[1]))
            for ch in range(len(channels)):
                inttmp=0
                if len(p)==2: inttmp=float(img[ch][p[0]][p[1]])
                if len(p)==3: inttmp=float(img[ch][p[0]][p[1]][p[2]])
                single_pixels_inside['intensity'][channels[ch]].append(inttmp)
                all_pixels_inside['sum_intensity'][channels[ch]]+=inttmp
            if len(p)==3:
                single_pixels_inside['z'].append(int(p[2]))



        cs=plt.contour(mask0, [0.5],linewidths=1.2,  colors='red')
        contcoords = cs.allsegs[0][0]
        z_flag=False
        if contcoords.shape[1]==3: z_flag=True
        for p in contcoords:
            single_pixels_contour['x'].append(p[0])
            single_pixels_contour['y'].append(p[1])
            if z_flag:single_pixels_contour['z'].append(p[2])
            for ch in range(len(channels)):
                inttmp=0
                if not z_flag: inttmp=float(img[ch][int(p[0])][int(p[1])])
                if z_flag: inttmp=float(img[ch][int(p[0])][int(p[1])][int(p[2])])
                single_pixels_contour['intensity'][channels[ch]].append(inttmp)
                all_pixels_contour['sum_intensity'][channels[ch]]+=inttmp


        for ch in range(len(channels)):
            all_pixels_inside['mean_intensity'][channels[ch]]=np.mean(single_pixels_inside['intensity'][channels[ch]])
            all_pixels_inside['std_intensity'][channels[ch]]=np.std(single_pixels_inside['intensity'][channels[ch]])
            all_pixels_contour['mean_intensity'][channels[ch]]=np.mean(single_pixels_contour['intensity'][channels[ch]])
            all_pixels_contour['std_intensity'][channels[ch]]=np.std(single_pixels_contour['intensity'][channels[ch]])

        center={'x':contours[c].centroid[0],'y':contours[c].centroid[1],'z':0}
        if len(contours[c].centroid)==3:
            center={'x':contours[c].centroid[0],'y':contours[c].centroid[1],'z':contours[c].centroid[2]}


        all_pixels_inside['npixels']  = int(contours[c].num_pixels)
        all_pixels_contour['npixels'] = len(contcoords)
        
        contour_dic={
            'center':center,
            'number_of_pixels':contours[c].num_pixels,
            'single_pixels_inside':single_pixels_inside,
            'all_pixels_inside':all_pixels_inside,
            'single_pixels_contour':single_pixels_contour,
            'all_pixels_contour':all_pixels_contour,
        }
        out_contours.append(contour_dic)
        plt.figure().clear()
        plt.close()
        plt.cla()
        plt.clf()

    return out_contours


########################################################################################################
########################################################################################################
########################################################################################################
########################################################################################################


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
def customLocalThresholding_Segmentation_out(img,  outdir, thr=2., delta=1, npix_min=400, npix_max=4000):

    for im in range(len(img)):
        image=img[im][0]
        img_seeds=fastiter(image, thr, delta)

        #dilated = binary_dilation(img_seeds, disk(2))
        closed = binary_closing(img_seeds, disk(4))
        filled = binary_fill_holes(closed).astype(int)
        label_im = label(filled)

        regions=regionprops(label_im)
        cells=[]
        for r in regions:
            if r.area>npix_min and r.area<npix_max:
                cells.append(r)
    
        previous_labels=[]
        if im>0:
            previous_labels = glob.glob(os.path.join(outdir, 'mask_tf{}_thr{}delta{}_cell*.json'.format(im-1,thr,delta)))
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

            for i in range(len(img[im])):
                intensity=0
                intensity_list=[]
                for coord in cells[c].coords:
                    intensity+=img[im][i][coord[0]][coord[1]]
                    intensity_list.append(img[im][i][coord[0]][coord[1]])
                intensities.append(intensity)
                intensities_list.append(intensity_list)

            mask0=np.zeros(img[im][0].shape, dtype=bool)
            for coord in cells[c].coords:
                mask0[coord[0]][coord[1]]=True
            cs=plt.contour(mask0, [0.5],linewidths=1.2,  colors='red')
            contcoords= cs.allsegs[0][0]
        
            print('frame = ', im)
            print('area  = ',cells[c].area)
            print('num_pixels  = ',cells[c].num_pixels)
            

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
            outname=os.path.join(outdir, 'mask_tf{}_thr{}delta{}_{}.json'.format(im,thr,delta,celllabel))
            with open(outname, "w") as outfile:
                outfile.write(json_object)


        #Take simple seg as default
        celldic={}
        celllist=glob.glob(os.path.join(outdir , 'mask_tf{}_thr{}delta{}_cell*.json'.format(im,2.,2)))
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
        outnametf=os.path.join(outdir, 'metadata_tf{}.json'.format(im))
        if not os.path.isfile(outnametf):
            with open(outnametf, "w") as outfiletf:
                outfiletf.write(jsontf_object)


#_______________________________________________
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
def apoc_Segmentation(img, outdir, npix=300, model="models/pixel_classification.cl"):
    clf = apoc.PixelClassifier(opencl_filename=model)

    image=img[:,0,:,:]
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
            for i in range(len(img[count])):
                intensity=0
                intensity_list=[]
                for coord in cells[c].coords:
                    intensity+=img[count][i][coord[0]][coord[1]]
                    intensity_list.append(img[count][i][coord[0]][coord[1]])
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
                'nchannels':len(img[count]),
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
