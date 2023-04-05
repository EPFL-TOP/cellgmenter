import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import glob, os, sys
import matplotlib.patches as patches
import scipy.stats as stats
import json
import math
import segmentation as seg
import reader as myread
#np.set_printoptions(threshold=sys.maxsize)

#image
fig1 = matplotlib.figure.Figure(figsize=(5,5))
fig1.add_subplot(111).plot([],[])

#image cropped
fig5 = matplotlib.figure.Figure(figsize=(5,5))
fig5.add_subplot(111).plot([],[])

#intensity
fig2 = matplotlib.figure.Figure(figsize=(15,5))
fig2.add_subplot(111).plot([],[])

#gradient
fig3 = matplotlib.figure.Figure(figsize=(5,5))
fig3.add_subplot(211).plot([],[])
fig3.add_subplot(212).plot([],[])

#histogram
fig4 = matplotlib.figure.Figure(figsize=(10,5))
fig4.add_subplot(121).plot([],[])
fig4.add_subplot(122).plot([],[])

#intensity for all positions
fig6 = matplotlib.figure.Figure(figsize=(20,10))


#use windows or linux path here
path = "/Users/helsens/data/singleCell"
path = "C:\Users\Laurel\software\cellgmenter"
metadatapath=os.path.join(path, "metadata")


theimagemeta=''
count=0

cellscolors=['b','r','g','c','m','y','k','w']
cellsmarkers=['x','o','.']
currentcell='cell0'


def crop(image, x1, x2, y1, y2):
    """
    Return the cropped image at the x1, x2, y1, y2 coordinates
    """
    if x2 == -1:
        x2=image.shape[1]-1
    if y2 == -1:
        y2=image.shape[0]-1

    mask = np.zeros(image.shape)
    mask[y1:y2+1, x1:x2+1]=1
    m = mask>0

    return image[m].reshape((y2+1-y1, x2+1-x1))

#_______________________________________________
def update_crop(img, x1, x2, y1, y2, mask_name):

	axes5 = fig5.axes
	axes5[0].cla()
	image_cropped = crop(img[0], x1, x2, y1, y2)
	axes5[0].imshow(image_cropped, cmap='gray')

	mask0=np.zeros(img[0].shape, dtype=bool)
	outnames=glob.glob(os.path.join(theimagemeta, 'mask_tf{}_{}_cell*.json'.format(count, mask_name)))
	for c in range(len(outnames)):
		f = open(outnames[c])
		data = json.load(f)
		for coord in data['coords']:
			mask0[coord[0]][coord[1]]=True
	mask_cropped = crop(mask0, x1, x2, y1, y2)
	axes5[0].contour(mask_cropped, [0.5],linewidths=1.2,  colors='red')
	#axes5[0].imshow(mask_cropped, cmap='gray')



#_______________________________________________
def update_segmentation(img, seg_npix=400, seg_thr=2.5, seg_delta=1):
	outnames=glob.glob(os.path.join(theimagemeta,'mask_tf{}_thr{}delta{}_cell*.json'.format(count,seg_thr,seg_delta)))
	if len(outnames)==0:
		seg.simpleSeg(img, theimagemeta, count, thr=seg_thr, delta=seg_delta, npix=seg_npix)


#_______________________________________________
def update_figure(img, 
		  xg=0, yg=0, dxg=10, dyg=10, 
		  xb=100, yb=100, dxb=50, dyb=50, 
		  binsh=100, minh=-1, maxh=-1, logh=False, 
		  binshc=100, minhc=-1, maxhc=-1, loghc=False, 
		  ch0=True, ch1=True, ch2=True):

	axes1 = fig1.axes
	axes2 = fig2.axes
	axes3 = fig3.axes
	axes4 = fig4.axes

	axes1[0].cla()
	axes2[0].cla()
	axes3[0].cla()
	axes3[1].cla()
	axes4[0].cla()
	axes4[1].cla()

	masks=None
	md_path=os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))
	if not os.path.isfile(md_path):
		print('no metadata for: ',md_path, ' will crash :) ')
	
	md_file = open(md_path)
	md_data = json.load(md_file)
	for x in md_data['cells']:print(x)
	outnames=[md_data['cells'][x]['mask'] for x in md_data['cells']]
	print(outnames)
	if xg+dxg>=len(img[0])-1:dxg=len(img[0])-1-xg
	if yg+dyg>=len(img[0])-1:dyg=len(img[0])-1-yg

	recth = patches.Rectangle((0, yg), len(img[0])-1, dyg, linewidth=1, edgecolor='r', facecolor='none')
	rectv = patches.Rectangle((xg, 0), dxg, len(img[0])-1, linewidth=1, edgecolor='b', facecolor='none')

	if xb+dxb>=len(img[0])-1:dxb=len(img[0])-1-xb
	if yb+dyb>=len(img[0])-1:dyb=len(img[0])-1-yb

	rectb = patches.Rectangle((xb, yb), dxb, dyb, linewidth=1, edgecolor='black', facecolor='none')

	meanv=[]
	meanh=[]
	stdv=[]
	stdh=[]

	#Horizontal gradient (scan at a given y in x values)
	for i in range(len(img[0])):
		tmp=img[0][yg:yg+dyg, i:i+1].flatten()
		meanh.append(np.mean(tmp))
		stdh.append(np.std(tmp))

	#Vertical gradient (scan at a given x in y values)
	for i in range(len(img[0])):
		tmp=img[0][i:i+1, xg:xg+dxg].flatten()
		meanv.append(np.mean(tmp))
		stdv.append(np.std(tmp))


	boxint=img[0][yb:yb+dyb, xb:xb+dxb].flatten()

	axis=[i for i in range(len(img[0]))]

	axes1[0].imshow(img[0], cmap='gray')#, vmin=0, vmax=2550)
	axes1[0].add_patch(recth)
	axes1[0].add_patch(rectv)
	axes1[0].add_patch(rectb)


	if md_data['skipframe']=="True":
		axes1[0].text(0.25, 0.1, 'SKIP FRAME',  fontsize=15, horizontalalignment='center', verticalalignment='center', color='white', transform=axes1[0].transAxes)

	for c in range(len(outnames)):
		f = open(outnames[c])
		data = json.load(f)
		validtxt=''
		if md_data['cells'][data['label']]['valid']=="False":validtxt=" not valid"
		if md_data['cells'][data['label']]['alive']=="True": axes1[0].text(data['center'][1], data['center'][0]-50, data['label']+validtxt, fontsize=10, horizontalalignment='center', verticalalignment='center', color='white')
		else:axes1[0].text(data['center'][1], data['center'][0]-50, data['label']+validtxt, fontsize=10, horizontalalignment='center', verticalalignment='center', color='black')
		axes1[0].scatter(data['center'][1], data['center'][0], color='white', marker="x", s=15) # plotting single point
		mask0=np.zeros(img[0].shape, dtype=bool)
		for coord in data['coords']:
			mask0[coord[0]][coord[1]]=True
		cs=axes1[0].contour(mask0, [0.5],linewidths=1.2,  colors='red')

	axes3[0].plot(axis,stdh,'r-', label="horizontal")
	axes3[0].plot(axis,stdv,'b-', label="vertical")
	axes3[0].legend(loc='upper left')
	axes3[0].set_ylabel('$\sigma$ intensity')

	axes3[1].plot(axis,meanh,'r-', label="horizontal")
	axes3[1].plot(axis,meanv,'b-', label="vertical")
	axes3[1].set_xlabel('pixel number')
	axes3[1].set_ylabel('Mean intensity')

	hmin=min(boxint)-10
	hmax=max(boxint)+10
	if minh>0:hmin=minh
	if maxh>0:hmax=maxh

	if logh:
		axes4[0].hist(boxint,bins=binsh, range=[hmin, hmax],log=True, density=True)		
	else:
		axes4[0].hist(boxint,bins=binsh, range=[hmin, hmax], density=True)

	axes4[0].set_xlabel('bright field intensity')
	axes4[0].text(0.025, 0.9,  'Mean:\t{:.1f}'.expandtabs().format(np.mean(boxint)), fontsize=10, horizontalalignment='left', transform=axes4[0].transAxes)
	axes4[0].text(0.025, 0.85, 'Std:\t{:.1f}'.expandtabs().format(np.std(boxint)), fontsize=10, horizontalalignment='left',  transform=axes4[0].transAxes)
	axes4[0].text(0.025, 0.8,  'Npix:\t  {}'.expandtabs().format(len(boxint)), fontsize=10, horizontalalignment='left',  transform=axes4[0].transAxes)

	xgaus = np.linspace(np.mean(boxint) - 3*np.std(boxint), np.mean(boxint) + 3*np.std(boxint), 100)
	axes4[0].plot(xgaus, stats.norm.pdf(xgaus, np.mean(boxint), np.std(boxint)))	
	
	cellmean='Mean:'
	cellstd='Std:'
	cellpix='Npix:'
	for c in range(len(outnames)):
		f = open(outnames[c])
		data = json.load(f)
		bfint=[img[0][d[0]][d[1]] for d in data['coords'] ]
		hminc=min(bfint)-10
		hmaxc=max(bfint)+10
		if minhc>0:hminc=minhc
		if maxhc>0:hmaxc=maxhc
		xgaus = np.linspace(np.mean(bfint) - 3*np.std(bfint), np.mean(bfint) + 3*np.std(bfint), 100)
		axes4[1].plot(xgaus, stats.norm.pdf(xgaus, np.mean(bfint), np.std(bfint)), color=cellscolors[c])	

		if loghc:
			axes4[1].hist(bfint,bins=binshc, density=True, color=cellscolors[c], alpha=0.5, label='cell{}'.format(c), log=True, range=[hminc, hmaxc])
		else:	
			axes4[1].hist(bfint,bins=binshc, density=True, color=cellscolors[c], alpha=0.5, label='cell{}'.format(c), range=[hminc, hmaxc])
		cellmean+=', {:.1f}'.format(np.mean(bfint))
		cellstd+=', {:.1f}'.format(np.std(bfint))
		cellpix+=', {}'.format(len(bfint))

		timelaps=glob.glob(outnames[c].replace('_tf{}_'.format(count), '*'))
		timelaps.sort()
		for ch in range(data['nchannels']):
			time=[]
			intensity=[]
			for tf in timelaps:
				f2 = open(tf)
				data2 = json.load(f2)
				intensity.append(data2['intensity'][ch]/data2['npixels'])
				
				timelist=os.path.split(tf)[-1].split('_')
				for t in timelist:
					if 'tf' in t: 
						time.append(10*int(t.replace('tf','')))


			zipped_lists = zip(time, intensity)
			sorted_pairs = sorted(zipped_lists)
			tuples = zip(*sorted_pairs)
			time, intensity = [ list(tuple) for tuple in  tuples]
			if ch==0 and ch0:axes2[0].plot(time, intensity, cellscolors[c]+cellsmarkers[ch]+'-', label="ch{}_cell{}".format(ch, c))
			if ch==1 and ch1:axes2[0].plot(time, intensity, cellscolors[c]+cellsmarkers[ch]+'-', label="ch{}_cell{}".format(ch, c))
			if ch==2 and ch2:axes2[0].plot(time, intensity, cellscolors[c]+cellsmarkers[ch]+'-', label="ch{}_cell{}".format(ch, c))
	axes2[0].set(xlabel='Time (minutes)', ylabel='Intensity/number of pixels')
	axes2[0].axvline(x = count*10, color = 'b', label = '')
	axes2[0].legend(loc='upper right')

	axes4[1].text(0.025, 0.9, cellmean, fontsize=10, horizontalalignment='left', transform=axes4[1].transAxes)
	axes4[1].text(0.025, 0.85, cellstd, fontsize=10, horizontalalignment='left',  transform=axes4[1].transAxes)
	axes4[1].text(0.025, 0.8, cellpix, fontsize=10, horizontalalignment='left',  transform=axes4[1].transAxes)

	axes4[1].legend(loc='upper right')


#_______________________________________________
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

#_______________________________________________
def update_celllist(window):
	cells_list=[]
	if os.path.isfile(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))):
		tf_file = open(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count)))
		tf_data = json.load(tf_file)
		for cell in tf_data['cells']:cells_list.append(cell)
	cells_list.sort()
	window['-CELLLIST-'].update(currentcell, values=cells_list)


#_______________________________________________
def update_intensity_summary(npos, timelaps):
	axes6 = fig6.axes
	for np in range(npos):
		axes6[np].cla()
	
	poslist=glob.glob(os.path.join(timelaps,os.path.split(timelaps)[-1]+"*"))

	for pos in range(len(poslist)):
		mask_list={}
		tf_list=glob.glob(os.path.join(poslist[pos],'metadata_tf*.json'))
		for tf in tf_list:
			f = open(tf)
			data = json.load(f)
			for cell in data['cells']:
				found=False
				for cell2 in mask_list:
					if cell==cell2:found=True
				if found:mask_list[cell].append(data['cells'][cell]['mask'])
				else: mask_list[cell]=[data['cells'][cell]['mask']]
		intensities={}
		for cell in mask_list:
			ch0_int=[]
			ch1_int=[]
			ch2_int=[]
			ch_tf=[]
			for mask in mask_list[cell]:
				fm = open(mask)
				datam = json.load(fm)
				for ch in range(int(datam["nchannels"])):
					if ch==0:ch0_int.append(datam["intensity"][ch]/datam["npixels"])
					if ch==1:ch1_int.append(datam["intensity"][ch]/datam["npixels"])
					if ch==2:ch2_int.append(datam["intensity"][ch]/datam["npixels"])
				ch_tf.append(int(os.path.split(mask)[-1].split("_")[1].replace('tf','')))
			
			zipped_lists = zip(ch_tf, ch0_int)
			sorted_pairs = sorted(zipped_lists)
			tuples = zip(*sorted_pairs)
			time, ch0_int= [ list(tuple) for tuple in  tuples]

			zipped_lists = zip(ch_tf, ch1_int)
			sorted_pairs = sorted(zipped_lists)
			tuples = zip(*sorted_pairs)
			time, ch1_int= [ list(tuple) for tuple in  tuples]

			zipped_lists = zip(ch_tf, ch2_int)
			sorted_pairs = sorted(zipped_lists)
			tuples = zip(*sorted_pairs)
			time, ch2_int= [ list(tuple) for tuple in  tuples]


			axes6[pos].plot(time, ch1_int, cellscolors[int(cell.replace('cell',''))]+'-', label="ch1_cell{}".format(cell.replace('cell','')))
			axes6[pos].plot(time, ch2_int, cellscolors[int(cell.replace('cell',''))]+'-', label="ch2_cell{}".format(cell.replace('cell','')))
			axes6[pos].text(0.5, 0.9, os.path.split(poslist[pos])[-1], fontsize=10, horizontalalignment='left', transform=axes6[pos].transAxes)
			intensities[cell]={'ch0':[ch0_int,time], 'ch1':[ch1_int,time], 'ch2':[ch2_int,time]}




timelaps_list = os.listdir(metadatapath)
timelaps_list = [x for x in timelaps_list if x[0]!='.' and 'metadata' not in x]
timelaps_list.sort()
print(timelaps_list)


AppFont = 'Helvetica 12'
#AppFont = 'Times New Roman 12'
TabFont = 'Helvetica 14'
sg.theme('DarkTeal12')
control_col = sg.Column([
    [sg.Text("Project", size=(10, 1), key='-TIMELAPSTEXT-', font=AppFont), sg.Combo(timelaps_list, enable_events=True,key='-TIMELAPSLIST-', size=(20, 1), font=AppFont)],
    [sg.Text("Position", size=(10, 1), key='-POSITIONTEXT-', font=AppFont), sg.Combo([],enable_events=True,key='-POSITIONSLIST-', size=(20, 1), font=AppFont)],
	[sg.Button('Previous', font=AppFont, key='-TIMEFRAMEPREVIOUS-'), sg.Button('Next', font=AppFont, key='-TIMEFRAMENEXT-'), sg.Text("0/0", size=(10, 1), key='-TIMEFRAMECOUNTER-', font=AppFont),sg.Button('GoTo', font=AppFont, key='-TIMEFRAMEGOTOBUTTON-'), sg.Input(default_text='0',key='-TIMEFRAMEGOTO-',  size =3, justification='left', font=AppFont)],
    [sg.Slider(range=(0,0), orientation='h',change_submits=True, key='-TIMEFRAMESLIDER-', font=AppFont)],
	[sg.Text("", size=(10, 2),  font=AppFont)],
	
	[sg.Submit('Segmentation',key='-SEGSUBMIT-', font=AppFont)],
    [
	sg.Text('Npix',  size =(4, 1), font=AppFont), sg.Input(default_text='400',key='-SEGNPIX-',  size =3, justification='left', font=AppFont), 
    sg.Text('Thr',   size =(3, 1), font=AppFont), sg.Input(default_text='2.',key='-SEGTHR-',   size =3, justification='left', font=AppFont),
    sg.Text('Delta', size =(5, 1), font=AppFont), sg.Input(default_text='2',  key='-SEGDELTA-', size =3, justification='left', font=AppFont),
	],


    [sg.Submit('Gradient',key='-GRADIENTSUBMIT-', font=AppFont)],
    [
	sg.Text('x',  size =(1, 1), font=AppFont), sg.Input(default_text='0',  key='-XGRADIENT-',  size =3, justification='left', font=AppFont), 
    sg.Text('dx', size =(2, 1), font=AppFont), sg.Input(default_text='10', key='-DXGRADIENT-', size =3, justification='left', font=AppFont),
    sg.Text('y',  size =(1, 1), font=AppFont), sg.Input(default_text='0',  key='-YGRADIENT-',  size =3, justification='left', font=AppFont),
    sg.Text('dy', size =(2, 1), font=AppFont), sg.Input(default_text='10', key='-DYGRADIENT-', size =3, justification='left', font=AppFont),
	],

 	[sg.Submit('Box',key='-BOXSUBMIT-', font=AppFont)],	
	[
	sg.Text('x',  size =(1, 1), font=AppFont), sg.Input(default_text='100', key='-XSTART-',size =3, justification='left', font=AppFont), 
	sg.Text('dx', size =(2, 1), font=AppFont), sg.Input(default_text='50',  key='-XDELTA-',size =3, justification='left', font=AppFont),
	sg.Text('y',  size =(1, 1), font=AppFont), sg.Input(default_text='100', key='-YSTART-',size =3, justification='left', font=AppFont),
	sg.Text('dy', size =(2, 1), font=AppFont), sg.Input(default_text='50',  key='-YDELTA-',size =3, justification='left', font=AppFont),
	],

 	[sg.Submit('Box Histogram',key='-HISTOBOXSUBMIT-', font=AppFont)],	
	[
	sg.Text('bins', size =(4, 1), font=AppFont), sg.Input(default_text='100', key='-HISTOBINSBOX-', size =3, justification='left', font=AppFont), 
	sg.Text('min',  size =(3, 1), font=AppFont), sg.Input(default_text='-1',  key='-HISTOMINBOX-',  size =3, justification='left', font=AppFont),
	sg.Text('max',  size =(3, 1), font=AppFont), sg.Input(default_text='-1',  key='-HISTOMAXBOX-',  size =3, justification='left', font=AppFont),
	sg.Checkbox('Log', key = '-HISTOLOGBOX-',enable_events=True)
	],

	[sg.Submit('Contour Histogram',key='-HISTOCONTSUBMIT-', font=AppFont)],	
	[
	sg.Text('bins', size =(4, 1), font=AppFont), sg.Input(default_text='100', key='-HISTOBINSCONT-', size =3, justification='left', font=AppFont), 
	sg.Text('min',  size =(3, 1), font=AppFont), sg.Input(default_text='-1',  key='-HISTOMINCONT-',  size =3, justification='left', font=AppFont),
	sg.Text('max',  size =(3, 1), font=AppFont), sg.Input(default_text='-1',  key='-HISTOMAXCONT-',  size =3, justification='left', font=AppFont),
	sg.Checkbox('Log', key = '-HISTOLOGCOUNT-',enable_events=True)
	],

	[sg.Text("Intensity plot channels", size=(20, 1), font=AppFont)],
	[
	sg.Checkbox('CH0', key = '-CHANNEL0-',enable_events=True, default=True, font=AppFont), 
	sg.Checkbox('CH1', key = '-CHANNEL1-',enable_events=True, default=True, font=AppFont), 
	sg.Checkbox('CH2', key = '-CHANNEL2-',enable_events=True, default=True, font=AppFont)
	],
	[sg.Text("", size=(10, 1),  font=AppFont)],

	[sg.Submit('Cell Metadata',key='-CELLMETADATASUBMIT-', font=AppFont)],	  
	[sg.Text("Cell", size=(10, 1), key='-CELLTEXT-', font=AppFont), sg.Combo([],enable_events=True,key='-CELLLIST-', size=(20, 1), font=AppFont)],
	[sg.Text("Mask: ", size=(10, 1), key='-MASKTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-MASKTEXT2-', font=AppFont), sg.Combo([],enable_events=True,key='-MASKLIST-', size=(10, 1), font=AppFont)],
	[sg.Text("Valid: ", size=(10, 1), key='-CHECKTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-CHECKTEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-CHECKLIST-', size=(10, 1), font=AppFont)],
	[sg.Text("Alive: ", size=(10, 1), key='-ALIVETEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-ALIVETEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-ALIVELIST-', size=(10, 1), font=AppFont)],
	[sg.Text("Status: ", size=(10, 1), key='-STATUSTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-STATUSTEXT2-', font=AppFont), sg.Combo(["single", "doublenuclei", "multiplecells", "pair"],enable_events=True,key='-STATUSLIST-', size=(10, 1), font=AppFont)],
	[sg.Text("Dividing: ", size=(10, 1), key='-DIVIDINGTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-DIVIDINGTEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-DIVIDINGLIST-', size=(10, 1), font=AppFont)],


	[sg.Submit('Timeframe Metadata',key='-TIMEFRAMEMETADATASUBMIT-', font=AppFont)],	 
	[sg.Text("Skipframe: ", size=(10, 1), key='-SKIPFRAMETEXT-', font=AppFont), sg.Text("False", size=(10, 1), key='-SKIPFRAMETEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-SKIPFRAMELIST-', size=(10, 1), font=AppFont)],

	[sg.Submit('Position Metadata',key='-POSITIONMETADATASUBMIT-', font=AppFont)],	 

	[sg.Submit('Project Metadata',key='-PROJECTMETADATASUBMIT-', font=AppFont)],	 


    ])
image_col = sg.Column([
	[sg.Canvas(key = '-CANVAS1-'), sg.Canvas(key = '-CANVAS2-')],
	[sg.Canvas(key = '-CANVAS5-'),sg.Canvas(key = '-CANVAS3-'), sg.Canvas(key = '-CANVAS4-')],
	])
layoutTF = [[control_col,image_col]]

image_col2 = [sg.Canvas(key = '-CANVAS6-')]
tab2=[image_col2]

layout = [[sg.TabGroup([
   [sg.Tab('Time frame', layoutTF, font=TabFont),
   sg.Tab('Time laps', tab2, font=TabFont)
   ]])]
]

window = sg.Window('Image Editor', layout, finalize = True, resizable=True, location=(100, 100))

firstimage=False
timelaps=''
position=''

fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)
fig6_agg = draw_figure(window['-CANVAS6-'].TKCanvas, fig6)

while True:
	position_list=[]

	event, values = window.read()#timeout=1000)
	print('event ',event)
	print('values ',values)
	if event!=None and values!=None:timeframeslider = int(values['-TIMEFRAMESLIDER-'])

	if event == sg.WIN_CLOSED: break
	

	if not values['-XGRADIENT-'].isnumeric() or not values['-YGRADIENT-'].isnumeric() or not values['-DXGRADIENT-'].isnumeric() or not values['-DYGRADIENT-'].isnumeric() :
		sg.popup_no_buttons('Value should be numeric', title="WARNING", font=TabFont)
		continue

	if event == '-SEGSUBMIT-':
		update_segmentation(image[count], seg_npix=int(values['-SEGNPIX-']), seg_thr=float(values['-SEGTHR-']), seg_delta=int(values['-SEGDELTA-']))

	if event == '-GRADIENTSUBMIT-' or event == '-HISTOLOGBOX-' or event == '-HISTOLOGCOUNT-' or \
		event == '-BOXSUBMIT-' or event=='-HISTOBOXSUBMIT-'  or event=='-HISTOCONTSUBMIT-' or \
		event == '-CHANNEL0-' or event =='-CHANNEL1-' or event=='-CHANNEL2-':
		print('---------update figure')
		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])
		
		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		update_celllist(window)


	if event == '-TIMELAPSLIST-':
		currentcell='cell0'

		timelaps=values['-TIMELAPSLIST-']
		project_meta=os.path.join(metadatapath, timelaps, 'project.json')
		if not os.path.isfile(project_meta): 
			print('no metadata for project , exit ',project_meta)
			sys.exit(3)
		projectfile_data = open(project_meta)
		project_data = json.load(projectfile_data)
		position_list_long=project_data['positions']
		position_list=[]
		for p in position_list_long:
			position_list.append(os.path.split(p)[-1].replace('.nd2','').replace('.tif',''))
		position_list.sort()	
		window['-POSITIONSLIST-'].update(values=position_list)

		fig6_agg.get_tk_widget().forget()
		plt.clf()
		xrow=int(math.sqrt(project_data['numberofpos']))+1
		for sp in range(1,project_data['numberofpos']+1):
			fig6.add_subplot(xrow,xrow,sp).plot([],[])
		update_intensity_summary(project_data['numberofpos'], os.path.join(metadatapath, timelaps))
		fig6_agg = draw_figure(window['-CANVAS6-'].TKCanvas, fig6)

	if event == '-POSITIONSLIST-':
		currentcell='cell0'
		position=values['-POSITIONSLIST-']
		position_meta=os.path.join(metadatapath, timelaps, position,'position.json')
		if not os.path.isfile(position_meta): 
			print('no metadata for position , exit ',position_meta)
			sys.exit(3)
		positionfile_data = open(position_meta)
		position_data = json.load(positionfile_data)
		
		theimagemeta=os.path.join(metadatapath, timelaps, position)
		#os.path.split(position_data['name'])[0]
		if not os.path.exists(theimagemeta):
			os.makedirs(theimagemeta)
		print('--------------------------------------theimagemeta',theimagemeta)


		if position_data['name'].split('.')[-1]=='nd2':
			image = myread.nd2reader(position_data['name'])
		if position_data['name'].split('.')[-1]=='tif':
			image = myread.tifreader(position_data['name'])
		nimage=len(image)
		print(image.shape)
		print('Nimages= ',len(image))
		window['-TIMEFRAMECOUNTER-'].update("{}/{}".format(1,nimage))
		window['-TIMEFRAMESLIDER-'].update(value=1, range=(1,nimage))
		print('---------first image')
		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])
		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)
		update_celllist(window)


	if count!=timeframeslider-1 and timeframeslider!=0:
		print('---------slider image')
		count=timeframeslider-1
		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)

		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])
		window['-TIMEFRAMECOUNTER-'].update("{}/{}".format(count+1,nimage))
		window['-TIMEFRAMESLIDER-'].update(count+1)
		update_celllist(window)

	if event == '-TIMEFRAMENEXT-':
		print('---------next image')
		count+=1
		if count>len(image)-1:count=len(image)-1
		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])

		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)

		window['-TIMEFRAMECOUNTER-'].update("{}/{}".format(count+1,nimage))
		window['-TIMEFRAMESLIDER-'].update(count+1)
		update_celllist(window)

	if event == '-TIMEFRAMEPREVIOUS-':
		print('---------previous image')
		count-=1
		if count<0:count=0		

		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])

		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)

		window['-TIMEFRAMECOUNTER-'].update("{}/{}".format(count+1,nimage))
		window['-TIMEFRAMESLIDER-'].update(count+1)
		update_celllist(window)



	if event == '-TIMEFRAMEGOTOBUTTON-':
		print('---------go to image')
		count=int(values['-TIMEFRAMEGOTO-'])-1
		if count<1:count=0
		if count>len(image)-1:count=len(image)-1		

		fig1_agg.get_tk_widget().forget()
		fig2_agg.get_tk_widget().forget()
		fig3_agg.get_tk_widget().forget()
		fig4_agg.get_tk_widget().forget()
		fig5_agg.get_tk_widget().forget()
		plt.clf()
		update_figure(image[count],
		xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
		xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
		binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
		binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
		ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])

		fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
		fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
		fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
		fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
		fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)

		window['-TIMEFRAMECOUNTER-'].update("{}/{}".format(count+1,nimage))
		window['-TIMEFRAMESLIDER-'].update(count+1)
		update_celllist(window)

	if event=='-CELLLIST-' or event=='-MASKLIST-':
		currentcell=values['-CELLLIST-']


	if event=='-MASKLIST-':
		fig5_agg.get_tk_widget().forget()
		mask = os.path.join(theimagemeta,'mask_tf{}_{}_{}.json'.format(count, values['-MASKLIST-'], values['-CELLLIST-']))
		if os.path.isfile(mask):
			mask_file = open(mask)
			mask_data = json.load(mask_file)
			center=mask_data['center']
			coords=mask_data['coords']
			xcoords=[x[0] for x in coords]
			ycoords=[x[1] for x in coords]

			update_crop(image[count], min(ycoords)-5, max(ycoords)+5,min(xcoords)-5, max(xcoords)+5, os.path.split(mask)[-1].split('_')[2])
			fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)


	if event=='-CELLLIST-' or event == '-CELLMETADATASUBMIT-': #event=='-MASKLIST-' or event=='-CHECKLIST-' or event=='-ALIVELIST-' or event=='-STATUSLIST-':
		if os.path.isfile(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))):
			tf_file = open(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count)))
			tf_data = json.load(tf_file)
			cell_data = tf_data["cells"][values['-CELLLIST-']]

			masklist=glob.glob(os.path.join(theimagemeta,'mask_tf{}_*_{}.json'.format(count, values['-CELLLIST-'])))
			masklist=[os.path.split(x)[-1].split('_')[2] for x in masklist]
			masklist.sort()
			window['-MASKLIST-'].update('',values=masklist)
			window['-CHECKLIST-'].update('')
			window['-ALIVELIST-'].update('')
			window['-STATUSLIST-'].update('')
			window['-DIVIDINGLIST-'].update('')

			window['-MASKTEXT2-'].update(os.path.split(tf_data["cells"][values['-CELLLIST-']]['mask'])[-1].split('_')[2])
			window['-CHECKTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['valid'])
			window['-ALIVETEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['alive'])
			window['-STATUSTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['status'])
			window['-DIVIDINGTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['isdividing'])

			if event == '-CELLMETADATASUBMIT-':
				if values['-MASKLIST-']!='':     tf_data["cells"][values['-CELLLIST-']]['mask']=os.path.join(theimagemeta,"mask_tf{}_{}_{}.json".format(count, values['-MASKLIST-'], values['-CELLLIST-']))
				if values['-CHECKLIST-']!='':    tf_data["cells"][values['-CELLLIST-']]['valid']=values['-CHECKLIST-']
				if values['-ALIVELIST-']!='':    tf_data["cells"][values['-CELLLIST-']]['alive']=values['-ALIVELIST-']
				if values['-STATUSLIST-']!='':   tf_data["cells"][values['-CELLLIST-']]['status']=values['-STATUSLIST-']
				if values['-DIVIDINGLIST-']!='': tf_data["cells"][values['-CELLLIST-']]['isdividing']=values['-DIVIDINGLIST-']

				jsontf_object = json.dumps(tf_data, indent=4)
				outnametf=os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))
				with open(outnametf, "w") as outfiletf:
					outfiletf.write(jsontf_object)

				window['-MASKTEXT2-'].update(os.path.split(tf_data["cells"][values['-CELLLIST-']]['mask'])[-1].split('_')[2])
				window['-CHECKTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['valid'])
				window['-ALIVETEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['alive'])
				window['-STATUSTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['status'])
				window['-DIVIDINGTEXT2-'].update(tf_data["cells"][values['-CELLLIST-']]['isdividing'])

			fig1_agg.get_tk_widget().forget()
			fig2_agg.get_tk_widget().forget()
			fig3_agg.get_tk_widget().forget()
			fig4_agg.get_tk_widget().forget()
			plt.clf()
			update_figure(image[count],
				xg=int(values['-XGRADIENT-']),     yg=int(values['-YGRADIENT-']),     dxg=int(values['-DXGRADIENT-']), dyg=int(values['-DYGRADIENT-']),
					xb=int(values['-XSTART-']),        yb=int(values['-YSTART-']),        dxb=int(values['-XDELTA-']),     dyb=int(values['-YDELTA-']),
					binsh=int(values['-HISTOBINSBOX-']),  minh=int(values['-HISTOMINBOX-']),    maxh=int(values['-HISTOMAXBOX-']),  logh=values['-HISTOLOGBOX-'],
					binshc=int(values['-HISTOBINSCONT-']),  minhc=int(values['-HISTOMINCONT-']),    maxhc=int(values['-HISTOMAXCONT-']),  loghc=values['-HISTOLOGCOUNT-'],
					ch0=values['-CHANNEL0-'], ch1=values['-CHANNEL1-'], ch2=values['-CHANNEL2-'])

			fig1_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
			fig2_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
			fig3_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
			fig4_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)


	if event == '-TIMEFRAMEMETADATASUBMIT-':
		window['-SKIPFRAMELIST-'].update('')
		if os.path.isfile(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))):
			tf_file = open(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count)))
			tf_data = json.load(tf_file)
			if values['-SKIPFRAMELIST-']!='': tf_data["skipframe"]=values['-SKIPFRAMELIST-']
			jsontf_object = json.dumps(tf_data, indent=4)
			outnametf=os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))
			with open(outnametf, "w") as outfiletf:
				outfiletf.write(jsontf_object)
			window['-SKIPFRAMETEXT2-'].update(tf_data["skipframe"])


	if os.path.isfile(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count))) and not event=='-MASKLIST-':
		tf_file = open(os.path.join(theimagemeta,'metadata_tf{}.json'.format(count)))
		tf_data = json.load(tf_file)
		window['-SKIPFRAMETEXT2-'].update(tf_data["skipframe"])
		if len(tf_data["cells"])>0:
			mask = tf_data["cells"][currentcell]['mask']
			if os.path.isfile(mask):
				mask_file = open(mask)
				mask_data = json.load(mask_file)
				center=mask_data['center']
				coords=mask_data['coords']
				xcoords=[x[0] for x in coords]
				ycoords=[x[1] for x in coords]
				fig5_agg.get_tk_widget().forget()
				update_crop(image[count], min(ycoords)-5, max(ycoords)+5,min(xcoords)-5, max(xcoords)+5, os.path.split(mask)[-1].split('_')[2])
				fig5_agg = draw_figure(window['-CANVAS5-'].TKCanvas, fig5)
				window['-MASKTEXT2-'].update(os.path.split(tf_data["cells"][currentcell]['mask'])[-1].split('_')[2])
				window['-CHECKTEXT2-'].update(tf_data["cells"][currentcell]['valid'])
				window['-ALIVETEXT2-'].update(tf_data["cells"][currentcell]['alive'])
				window['-STATUSTEXT2-'].update(tf_data["cells"][currentcell]['status'])
				window['-DIVIDINGTEXT2-'].update(tf_data["cells"][currentcell]['isdividing'])

	#[sg.Submit('TimeFrame Metadata',key='-TIMEFRAMEMETADATASUBMIT-', font=AppFont)],	  
	#[sg.Text("Cell", size=(10, 1), key='-CELLTEXT-', font=AppFont), sg.Combo([],enable_events=True,key='-CELLLIST-', size=(20, 1), font=AppFont)],
	#[sg.Text("Mask: ", size=(10, 1), key='-MASKTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-MASKTEXT2-', font=AppFont), sg.Combo([],enable_events=True,key='-MASKLIST-', size=(10, 1), font=AppFont)],
	#[sg.Text("Checked: ", size=(10, 1), key='-CHECKTEXT-', font=AppFont), sg.Text("False", size=(10, 1), key='-CHECKTEXT2-', font=AppFont), sg.Combo([],enable_events=True,key='-CHECKLIST-', size=(10, 1), font=AppFont)],
	#[sg.Text("Alive: ", size=(10, 1), key='-ALIVETEXT-', font=AppFont), sg.Text("True", size=(10, 1), key='-ALIVETEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-ALIVELIST-', size=(10, 1), font=AppFont)],
	#[sg.Text("Status: ", size=(10, 1), key='-STATUSTEXT-', font=AppFont), sg.Text("NA", size=(10, 1), key='-STATUSTEXT2-', font=AppFont), sg.Combo(["single", "doublenuclei", "multiplecells", "pair"],enable_events=True,key='-STATUSLIST-', #size=(10, 1), font=AppFont)],
	#[sg.Text("Dividing: ", size=(10, 1), key='-DIVIDINGTEXT-', font=AppFont), sg.Text("False", size=(10, 1), key='-DIVDINGTEXT2-', font=AppFont), sg.Combo(["True","False"],enable_events=True,key='-DIVIDINGLIST-', size=(10, 1), #font=AppFont)],




	if event == '-TIMEFRAMEMETADATA-':

		#For every cell in the time frame
		timeframedic={
			'mask':theimagemeta+'_tf{}_thr{}delta{}_cell*.json'.format(count,seg_thr,seg_delta),
			'hasbeenchecked':False, #Set to True when user has checked and validated the timeframe
			'alive':None, #True/False
			'status':'NA',  #'single, doublenuclei, multiplecells, pair'
			'isdividing':None, #True/False', can span over multiple TF
			'persistent':True, #True by default a cell that stays for the full timelaps
			'skipframe':False #false by default
	}	

	if event == '-TIMELAPSMETADATA-':
		#for every position
		positiondic={
			'experimentalcondition':'NA', #free field user specific
			'numberofposition':'',#auto from image shape
			'positions':'',
			'cellinfo':{'cell0':{'numbe of pics','time of first/last peak'}}
	}


		pprojectdic={
			
			'dissociationtime':None,#user specific
			'startofmovie':None#user specific
	}
window.close()




