from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

import os
import pandas as pd
import skfmm
from thefuzz import fuzz, process

import argparse

def geodesic_distance_transform(img, root):
    m = np.copy(img)
    m[root] = False
    m = np.ma.masked_array(m, ~img)
    return skfmm.distance(m, order=2).data

def read_binary_img(filename, threshold=100):
    bimg = cv2.imread(filename)[:,:,0]
    bimg[bimg < threshold] = 0
    bimg[bimg > 0] = 1
    bimg = bimg.astype(bool)

    return bimg

def get_largest_elements(comp, alpha=0.25):
    labels,num = ndimage.label(comp, structure=ndimage.generate_binary_structure(comp.ndim, 1))
    print(num,'components')
    hist, bins = np.histogram(labels, bins=num, range=(1,num+1))
    print(np.sort(hist)[::-1][:20])
    tot = np.sum(hist)

    box = np.zeros(comp.shape, dtype=bool)
    for i in np.nonzero(hist/tot > alpha)[0]:
        box = box | (labels == i+1)
    print('Returned only the largest',i+1,'components')
    return box

genotypes = ['CAL','MLB','222','299','517','521']
imtype = ['Diseased', 'Healthy', 'Binary']
fs = 15

def main():

    parser = argparse.ArgumentParser(description='Get geodesic distances from crown')
    parser.add_argument('runnum', metavar='run_number', type=int, help='directory where raw images are located')
    parser.add_argument('gidx', metavar='gene_id', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    runnum = args.runnum

    src = '../run{:02d}/'.format(runnum)

    psrc = src + 'processed/'
    ddst = src + 'diagnostic/'
    hdst = src + 'histograms/'
    
    graydots = glob('../Graydot_images_run{:02d}/*.jpg'.format(runnum))
    graydots = [ os.path.split(graydots[i])[1].split('.jpg')[0] for i in range(len(graydots)) ]
    
    stemcorrected = glob('../Graydot_images_run{:02d}/StemCorrected/*.jpg'.format(runnum))
    stemcorrected = [ os.path.split(stemcorrected[i])[1].split('.jpg')[0] for i in range(len(stemcorrected)) ]
        
    if not os.path.isdir(hdst):
        os.mkdir(hdst)

    gidx = args.gidx
    bfiles = sorted(glob(psrc + '*{}*.npy'.format(genotypes[gidx])))
    print('Total number of files:\t{}\n'.format(len(bfiles)))
    
    for idx in range(len(bfiles)):
        
        bname = os.path.splitext(os.path.split(bfiles[idx])[1])[0].split('_-_')[0]
        fmatch = process.extractOne(bname, choices=graydots, scorer=fuzz.partial_ratio)
        print(fmatch)
        
        if fmatch[1] > 90:
        
            ceros = np.array(os.path.splitext(bfiles[idx])[0].split('_')[-4:], dtype=int)
            zeroslice = np.s_[ceros[2]:ceros[3], ceros[0]:ceros[1]]

            print(bfiles[idx], bname, sep='\n')
            gimg = np.load(bfiles[idx], allow_pickle=True)

            
            filename = glob('../Graydot_images_run{:02d}/{}*.jpg'.format(runnum, fmatch[0]))
            bimg = cv2.imread(filename[0])[zeroslice][:,:,0]
            crownmask = (bimg > 100) & (bimg < 200)
            crown = ndimage.center_of_mass(crownmask)
            crown = tuple(np.array(crown).astype(int))
                
            stem = np.zeros_like(gimg)
            
            if len(stemcorrected) > 0:
                smatch = process.extractOne(bname, choices=stemcorrected, scorer=fuzz.partial_ratio)
                print(smatch)
                if smatch[1] > 90:
                    filename = glob('../Graydot_images_run{:02d}/StemCorrected/{}*.jpg'.format(runnum, smatch[0]))
                    simg = cv2.imread(filename[0])[zeroslice][:,:,0]
                    stem = get_largest_elements( (simg > 20) & (simg < 50) , alpha = 0.1)
                    
            rmask = gimg.copy()
            rmask[stem] = False

            # # Make a proper geodesic watershed

            rootbase = ( 0 , int(ndimage.center_of_mass(gimg[0])[0]) )
            cgdt = geodesic_distance_transform(gimg, crown)
            bgdt = geodesic_distance_transform(gimg, rootbase)
            rstem = bgdt < bgdt[crown]
            rstem[~gimg] = False
            b2c = np.sqrt(np.sum(np.power([crown[i] - rootbase[i] for i in range(len(crown))], 2)))
            if len(cgdt[rstem])/np.sum(cgdt != 0) < 0.75:
                cgdt[rstem] *= -1

            img = [ (read_binary_img(glob(src + '{}*/*{}*'.format(imtype[i], bname.replace('-','*')))[0], 25)[zeroslice])*rmask for i in range(2) ]
            cgeod = [ cgdt*img[i] for i in range(len(img)) ]

            # Histogram of pixel-geodesic distribution

            gvals = cgdt[rmask]
            vmin = int(np.floor(np.min(gvals)))
            vmax = int(np.ceil(np.max(gvals)))
            bins=range(vmin,vmax+1)
            
            for i in range(len(cgeod)):
                hist, _ = np.histogram(cgeod[i][img[i]], bins=bins)
                filename = hdst + bname + '_-_{}_{}_crown_geodesic.csv'.format(bins[-1], imtype[i].lower())
                print(filename)
                np.savetxt(filename, hist, delimiter=',', fmt='%d')
                    
    return 0

if __name__ == '__main__':
    main()
