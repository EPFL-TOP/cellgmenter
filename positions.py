import json
import glob
import os
from scipy.signal import find_peaks
import numpy as np

path='/Users/helsens/data/singleCell/metadata/*'

#---------------------------------------
def main():
	poslist=get_poslist()
	for pos in poslist:
	   process_pos(pos)
	   

#---------------------------------------
def get_ordered_list(l1, l2):
    zipped_lists = zip(l1, l2)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    return [ list(tuple) for tuple in  tuples]

#---------------------------------------
def process_pos(pos):   
    print('position ',pos)
    metadatalist=glob.glob(os.path.join(pos,'metadata_tf*.json'))
    metadatalist.sort()
    print('N frames ',len(metadatalist))
    mask_list={}

    for tf in metadatalist:
        f = open(tf)
        data = json.load(f)
        for cell in data['cells']:
            found=False
            for cell2 in mask_list:
                if cell==cell2:found=True
            if found:mask_list[cell].append(data['cells'][cell]['mask'])
            else: mask_list[cell]=[data['cells'][cell]['mask']]
    intensities={}
    posdata_dic={}
    for cell in mask_list:
        ch0_int=[]
        ch1_int=[]
        ch2_int=[]
        ch_tf=[]
        center=[]
        for mask in mask_list[cell]:
            fm = open(mask)
            datam = json.load(fm)
            for ch in range(int(datam["nchannels"])):
                if ch==0:ch0_int.append(datam["intensity"][ch]/datam["npixels"])
                if ch==1:ch1_int.append(datam["intensity"][ch]/datam["npixels"])
                if ch==2:ch2_int.append(datam["intensity"][ch]/datam["npixels"])
            ch_tf.append(int(os.path.split(mask)[-1].split("_")[1].replace('tf','')))
            center.append(datam["center"])


        time_int, ch0_int = get_ordered_list(ch_tf, ch0_int)
        time_int, ch1_int = get_ordered_list(ch_tf, ch1_int)
        time_int, ch2_int = get_ordered_list(ch_tf, ch2_int)
        time_int, center = get_ordered_list(ch_tf, center)

        peaksmax, _ = find_peaks(np.array(ch1_int),  prominence=40)
        peaksmin, _ = find_peaks(-np.array(ch1_int), prominence=40)

        peaksmax_list=[]
        peaksmin_list=[]
        if len(peaksmax)>0:peaksmax_list.append(int(peaksmax[0]))
        if len(peaksmin)>0:peaksmin_list.append(int(peaksmin[0]))

        for max in range(1, len(peaksmax)):
            if peaksmax[max]-peaksmax[max-1]<15: peaksmax_list.append(int(peaksmax[max]))
        for min in range(1, len(peaksmin)):
            if peaksmin[min]-peaksmin[min-1]<15: peaksmin_list.append(int(peaksmin[min]))

        oscilations_start=-1
        oscilations_end=-1
        if len(peaksmax_list)>0:
            for osc in range(peaksmax_list[-1], len(metadatalist)-5):
                print(osc, len(metadatalist), np.std([ch1_int[osc], ch1_int[osc+1], ch1_int[osc+2]]))
                if np.std([ch1_int[osc], ch1_int[osc+1], ch1_int[osc+2], ch1_int[osc+4], ch1_int[osc+5]])<10:
                    oscilations_end=osc
                    break



        #rising=[i for i in range(peaksmin_list[0]+1, peaksmax_list[-1]-1) if in not in peaksmin_list and i not in peaksmax_list]
        #falling=[i for i in range(peaksmax_list[0]+1, peaksmax_list[-1]-1) if in not in peaksmin_list and i not in peaksmax_list]

        rising=[]
        falling=[]
        #for min in range(peaksmin_list[0]+1):


        #add oscend
        #peaksmin_list.append(oscilations_end)
        #for tf in range(len(time_int)):
        #    for min in range(len(peaksmin_list)-1):
        #        for max in range(len(peaksmax_list)):
        #            if peaksmin_list[min]>peaksmax_list[max]:continue
        #            if peaksmax_list[max]>peaksmin_list[min+1]:continue
        #            #print('tf=',tf,'  min=',peaksmin_list[min],'  max=',peaksmax_list[max] )
        #            if tf>peaksmin_list[min] and tf<peaksmax_list[max]:
        #                rising.append(tf)


        #for tf in range(len(time_int)):
        #    if tf in rising:continue
        #    for max in range(len(peaksmax_list)):
        #        for min in range(len(peaksmin_list)-1):
        #            if peaksmin_list[min]<peaksmax_list[max]:continue
        #            #if peaksmax_list[max]<peaksmin_list[min+1]:continue
        #            if tf<peaksmin_list[min+1] and tf>peaksmax_list[max] and tf<oscilations_end :
        #                falling.append(tf)
        ##remove last item
        #peaksmin_list.pop()
        print(peaksmin, peaksmin_list)
        print(peaksmax, peaksmax_list)

        peaksmin_list_clean=[]
        for min in peaksmin_list:
            if min>peaksmax_list[-1]:continue
            if min<oscilations_end:peaksmin_list_clean.append(min)
        if len(peaksmin_list_clean)==len(peaksmax_list):
            #rising when TS start with minimum
            for i in range(len(peaksmin_list_clean)):
                for tf in range(len(time_int)):
                    if tf>peaksmin_list_clean[i] and tf<peaksmax_list[i]:rising.append(tf)

            #Falling when TS start with minimum
            peaksmin_list_clean.append(oscilations_end)
            for i in range(len(peaksmin_list_clean)-1):
                for tf in range(len(time_int)):
                    if tf<peaksmin_list_clean[i+1] and tf>peaksmax_list[i]:falling.append(tf)
            ##remove last item
            peaksmin_list_clean.pop()

        #start with a maximum
        elif len(peaksmin_list_clean)<len(peaksmax_list):
            peaksmin_list_clean.append(oscilations_end)
            for i in range(len(peaksmin_list_clean)):
                for tf in range(len(time_int)):
                    if tf<peaksmin_list_clean[i] and tf>peaksmax_list[i]:falling.append(tf)

            for i in range(len(peaksmin_list_clean)-1):
                for tf in range(len(time_int)):
                    if tf>peaksmin_list_clean[i] and tf<peaksmax_list[i+1]:rising.append(tf)
            peaksmin_list_clean.pop()

        else:
            print('why are you here,exit')   
            print('nmin: ',len(peaksmin_list_clean),'  nmax: ', len(peaksmax_list))
            sys.exit(3)
        print('rising  ',rising)
        print('falling ',falling)
        with open(os.path.join(pos,'position_data.json'), "w") as outfile_posdata:
            posdata_dic[cell]={
                'intensities':[ch0_int, ch1_int, ch2_int],
                'time':time_int,
                'maxima':peaksmax_list,
                'minima':peaksmin_list_clean,
                'oscilations_start':None,
                'oscilations_end':oscilations_end,
                'rising':rising,
                'falling':falling,
                'center':center
            }
            #print(posdata_dic)
            json_object = json.dumps(posdata_dic)
            outfile_posdata.write(json_object)
            outfile_posdata.close()


#---------------------------------------
def get_poslist():
    poslist=[]
    projects=glob.glob(path)
    for pr in projects:
        print('project ',pr)
        positions=glob.glob(os.path.join(path,pr,'*'))
        positions.sort()
        for pos in positions:
            if '.json' in pos: continue
            poslist.append(pos)
    return poslist




if __name__ == "__main__":
    main()

