import segmentation as seg
import reader as myread
import os
import glob
import json
import apoc
import argparse


testclement=True

#use windows or linux path here
path ="/Users/helsens/data/singleCell"

#path = r"E:\Laurel\WSC\NIS split multipoints"  

path_meta=os.path.join(path, "metadata")
if not os.path.exists(path_meta):
	os.makedirs(path_meta)
	
project_list = os.listdir(path)
project_list = [x for x in project_list if x[0]!='.' and 'metadata' not in x]
project_list.sort()
print('project list ',project_list)

clf = apoc.PixelClassifier(opencl_filename="pixel_classification.cl")

parser = argparse.ArgumentParser()
parser.add_argument("--project", help="project", type=str)
parser.add_argument("--position", help="project", type=str)
args = parser.parse_args()
if args.project!=None: project_list=[args.project]


for proj in project_list:
    #if testclement and ('060' not in proj and 'rd416' not in proj):continue
    if testclement and ('rd416' not in proj):continue
    position_list=[]
    for p in glob.glob(os.path.join(path, proj, '*')):
        if 'metadata'not in p: position_list.append(p)
    position_list.sort()
    print('position list ' ,position_list)
    
    #check if for a given project the metadata exist
    project_dir=os.path.join(path_meta, proj)
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
    project_meta = os.path.join(path_meta, proj, 'project.json')
    if not os.path.isfile(project_meta):
        with open(project_meta, "w") as outfile_proj:
            project_dic={
              'name':proj,
                'date':'changeme',
                'microscope':'changeme',
                'author':'changeme',
                'contributors':'changeme',
                'numberofpos':len(position_list),
                'positions':position_list
            }   
            json_object = json.dumps(project_dic, indent=4)
            outfile_proj.write(json_object)
            outfile_proj.close()

    #Load the project metadata 
    projectfile_data = open(project_meta)
    project_data = json.load(projectfile_data)

    #Check if all the positions have metadata
    for pos in position_list:
        if args.position not in pos: 
            print ('skip position ',pos)
            continue
        print('positon = ', pos )

        position_dir=os.path.join(path_meta, proj, os.path.split(pos)[-1].replace('.nd2','').replace('.tif',''))
        #position_dir=os.path.join(path_meta, proj, pos.split('/')[-1].replace('.nd2','').replace('.tif',''))
        if not os.path.exists(position_dir):
            os.makedirs(position_dir)
        elif testclement and os.path.exists(position_dir):  continue

        position_meta = os.path.join(position_dir, 'position.json')
        if not os.path.isfile(position_meta):
            with open(position_meta, "w") as outfile_pos:
                position_dic={
                    'name':pos,
			        'experimentalcondition':'changeme',
			        'dissociationtime':'changeme',
			        'startofmovie':'changeme',
                    'numberofframes':-1,
	            }
    
                json_object = json.dumps(position_dic, indent=4)
                outfile_pos.write(json_object)
                outfile_pos.close()

        #Load the position metadata 
        positionfile_data = open(position_meta)
        position_data = json.load(positionfile_data)
        if position_data['name'] not in project_data['positions']:
            project_data['positions'].append(position_data['name'])
            project_data['numberofpos']=len(project_data['positions'])


        #loop over images of each position
        image=None
        if   '.nd2' in os.path.split(position_data['name'])[-1]: image = myread.nd2reader(position_data['name'])
        elif '.tif' in os.path.split(position_data['name'])[-1]: image = myread.tifreader(position_data['name'])
        else: print('unsupported image format')
        print('simple seg')
        for img in range(len(image)):
            timeframe_meta=glob.glob(os.path.join(position_dir+'mask_tf{}_thr{}delta{}_cell*.json'.format(img,2.,2)))
            if len(timeframe_meta)==0:
                seg.simpleSeg(image[img], position_dir, img, thr=2., delta=2, npix=400)
                #seg.simpleSeg(image[img], position_dir, img, thr=2., delta=1, npix=400)
                #seg.simpleSeg(image[img], position_dir, img, thr=3., delta=1, npix=400)
                
        print('apoc seg')
        seg.apocSeg(clf, image, position_dir, npix=400)

    with open(project_meta, "w") as outfile_proj:
        project_data['positions'].sort()
        json_object = json.dumps(project_data, indent=4)
        outfile_proj.write(json_object)
        outfile_proj.close()