import sys, os, glob
import matplotlib.pyplot as plt

sys.path.append('/Users/helsens/Software/github/super-resolution')
from model import resolve_single
from model.edsr import edsr

from utils import load_image, plot_sample

outtraining='/Users/helsens/data/singleCell/training_highres/'

model = edsr(scale=4, num_res_blocks=16)
model.load_weights('/Users/helsens/Software/github/super-resolution/weights/edsr-16-x4/weights.h5')

#for label in ['maxima', 'minima', 'rising', 'falling']:
for label in ['osc','no_osc']:
    images=glob.glob(os.path.join('/Users/helsens/data/singleCell/training', label,'*'))
    print(images)
    for img in images:
        lr = load_image(img)
        sr = resolve_single(model, lr)
        outname=os.path.join(outtraining, label, img.split('/')[-1])
        #print(sr.shape)
        #sr_test=sr[:,:,0]
        #print(sr_test.shape)
        #io.imsave(outname, sr_test)
        #sys.exit()
        plt.imsave(outname, sr.numpy(), cmap='gray')

