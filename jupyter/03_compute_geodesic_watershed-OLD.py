from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage, spatial

import os
import pandas as pd
import skfmm

import gudhi as gd
import argparse

def read_binary_img(filename, threshold=100):
    bimg = cv2.imread(filename)[:,:,0]
    bimg[bimg < threshold] = 0
    bimg[bimg > 0] = 1
    bimg = bimg.astype(bool)

    return bimg

def pers2numpy(pers):
    bd = np.zeros((len(pers), 3), dtype=float)
    for i in range(len(bd)):
        bd[i, 0] = pers[i][0]
        bd[i, 1:] = pers[i][1]
    return bd
    
def root_watershed(inv, gimg, main, rest, tips, tmask):
    watershed = np.zeros(inv.shape, dtype=np.uint8)
    colorval = 1
    colordict = dict()
    
    for ix in range(len(rest[tmask])-1, -1, -1):
        print('Iteration:\t', ix)
        thr = main[rest[tmask][ix,1]]
        timg = gimg*(inv < thr)
        
        label, nums = ndimage.label( timg, structure=ndimage.generate_binary_structure(2,1))
        print('Found',nums,'connected components')
        hist, bins = np.histogram(label, bins=range(1,nums+2))
        hargsort = np.argsort(hist)[::-1]
        mainlabels = np.zeros(ix+1, dtype=int)
        for i in range(len(mainlabels)):
            foo = label[tuple(tips[tmask][i])]
            mainlabels[i] = foo-1
        mainlabels = mainlabels[mainlabels > -1]
        
        for i in range(len(mainlabels)):
            mask = (watershed == 0) & (label == mainlabels[i] + 1)
            watershed[mask] = colorval
            
            colorlist = []
            for j in range(len(tips[tmask])):
                if label[tuple(tips[tmask][j])] == mainlabels[i] + 1:
                    colorlist.append(rest[tmask][j,0])
            colordict[colorval] = colorlist
            colorval += 1

    return watershed, colordict

def get_birthdeath(img, vmax, thr=None):

    if thr is None:
        thr = (np.max(img) - np.min(img))/20
        
    BirthDeath = []
    
    label, nums = ndimage.label(img, structure=ndimage.generate_binary_structure(2,1))
    extrema = ndimage.extrema(img, label, index=range(1,nums+1))
    
    oslices = ndimage.find_objects(label)
    print('Found',nums,'connected components')

    lifespan = extrema[1] - extrema[0]
    lmask = lifespan > thr
    lzero = np.nonzero(lmask)[0]
    print('Thr:', np.round(thr), ':', len(lifespan), '--->', len(lzero), sep='\t')

    for k in range(len(lzero)):
        box = img[oslices[lzero[k]]].copy()
        box[label[oslices[lzero[k]]] != lzero[k] + 1] = 0
        inv = extrema[1][lzero[k]] - box
        
        cc = gd.CubicalComplex(top_dimensional_cells = inv)
        pers = cc.persistence(homology_coeff_field=2, min_persistence=10)

        bd = pers2numpy(pers)
        bd = np.atleast_2d(bd[np.all(bd < np.inf, axis=1), :])
        bd = bd[bd[:,0] == 0, 1:] + (vmax - extrema[1][lzero[k]])
        bd = np.vstack(([ vmax - extrema[1][lzero[k]] , vmax - extrema[0][lzero[k]] ], bd))
        BirthDeath.append(bd)

    BirthDeath.append(np.column_stack((extrema[0][~lmask & (lifespan > 10)], extrema[1][~lmask & (lifespan > 10)])))
    
    return BirthDeath

genotypes = ['CAL','MLB','222','299','517','521']
imtype = ['Diseased', 'Healthy', 'Binary']
fs = 15

def main():

    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('runnum', metavar='raw_dir', type=int, help='directory where raw images are located')
    parser.add_argument('gidx', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    runnum = args.runnum

    src = '../run{:02d}/'.format(runnum)

    psrc = src + 'processed/'
    gsrc = src + 'gudhi/'
    ddst = src + 'diagnostic/'
    hdst = src + 'histograms/'

    if not os.path.isdir(hdst):
        os.mkdir(hdst)

    gidx = args.gidx
    bfiles = sorted(glob(psrc + '*{}*.npy'.format(genotypes[gidx])))
    print('Total number of files:\t{}\n'.format(len(bfiles)))
    
    for idx in range(len(bfiles)):
        bname = os.path.splitext(os.path.split(bfiles[idx])[1])[0].split('_-_')[0]
        ceros = np.array(os.path.splitext(bfiles[idx])[0].split('_')[-4:], dtype=int)
        zeroslice = np.s_[ceros[2]:ceros[3], ceros[0]:ceros[1]]

        print(bfiles[idx], bname, sep='\n')
        gimg = np.load(bfiles[idx], allow_pickle=True)

        # # Make a proper geodesic watershed

        m = np.copy(gimg)
        m[0, gimg[0] ] = False
        m = np.ma.masked_array(m, ~gimg)

        gdt = skfmm.distance(m).data
        inv = np.max(gdt) - gdt
        main = np.ravel(inv, 'F')

        filename = gsrc + bname + '_-_binary_H0.csv'
        birthdeath = pd.read_csv(filename)
        tips = birthdeath.loc[:, ['tipX','tipY']].values
        merge = birthdeath.loc[:, ['endX', 'endY']].values
        rest = birthdeath.loc[:, ['tipF','endF']].values

        filename = gsrc + bname + '_-_root_tips.csv'
        tmask = np.zeros(len(birthdeath), dtype=bool)
        tmask[ np.atleast_1d(np.loadtxt(filename, delimiter=',', dtype=int)) ] = True
        print('Found ', np.sum(tmask), ' root tips')

        watershed, colordict = root_watershed(inv, gimg, main, rest, tips, tmask)

        wshds = np.zeros((np.sum(tmask), gimg.shape[0], gimg.shape[1])) 
        for i in range(len(wshds)):
            m = np.zeros_like(gimg)
            for key in colordict:
                if rest[tmask, 0][i] in colordict[key]:
                    m[watershed == key] = True
            foo = np.copy(m)
            m[tuple(tips[tmask][i])] = False
            m = np.ma.masked_array(m, ~foo)
            wshds[i] = skfmm.distance(m).data

            wshds[i][wshds[i] <= 0] = -2**16
            wshds[i][tuple(tips[tmask][i])] = 0

        vmax = np.max(wshds)
        wshds = np.abs(wshds)
        rootw = np.min(wshds, axis=0)
        rootw[rootw > 2**15] = 0
        
        foo = np.sum( (gdt > 0) != (rootw > 0) )
        if foo > 200:
            print('EXAMINE THE ROOTW ARRAY FURTHER', bname)
            print('There are',foo,'missing pixels')
        
        else:
            # # Compare healthy from diseased

            rimg = [ (read_binary_img(glob(src + '{}*/*{}*'.format(imtype[i], bname.replace('-','*')))[0], 25)[zeroslice])*gimg for i in range(2) ]
            rgeod = [ gdt*rimg[i] for i in range(len(rimg)) ]
            tgeod = [ rootw*rimg[i] for i in range(len(rimg)) ]

            # Histogram of pixel-geodesic distribution

            vmax = int(np.round(np.max(gdt),0))
            for i in range(len(rgeod)):
                filename = hdst + bname + '_-_base_geodesic_{}.csv'.format(imtype[i].lower())
                if not os.path.isfile(filename):
                    hist, bins = np.histogram(rgeod[i], bins=range(1,vmax+2))
                    print(filename)
                    np.savetxt(filename, hist, delimiter=',', fmt='%d')

            vmax = int(np.round(np.max(rootw),0))
            for i in range(len(tgeod)):
                filename = hdst + bname + '_-_tips_geodesic_{}.csv'.format(imtype[i].lower())
                if not os.path.isfile(filename):
                    hist, bins = np.histogram(tgeod[i], bins=range(1,vmax+2))
                    print(filename)
                    np.savetxt(filename, hist, delimiter=',', fmt='%d')

            # H0 persistence for each of the geodesic filters

            vmax = np.round(np.max(gdt),0)
            for i in [0,1]:
                filename = gsrc + bname + '_-_base_geodesic_{}_H0.csv'.format(imtype[i].lower())
                if not os.path.isfile(filename):
                    BirthDeath = get_birthdeath(rgeod[i], vmax)
                    print(filename)
                    np.savetxt(filename, np.round(np.vstack(BirthDeath)).astype(int), delimiter=',', fmt='%d')

            vmax = np.round(np.max(rootw),0)
            for i in [0,1]:
                filename = gsrc + bname + '_-_tips_geodesic_{}_H0.csv'.format(imtype[i].lower())
                if not os.path.isfile(filename):
                    BirthDeath = get_birthdeath(tgeod[i], vmax)
                    print(filename)
                    np.savetxt(filename, np.round(np.vstack(BirthDeath)).astype(int), delimiter=',', fmt='%d')
                    
    return 0

if __name__ == '__main__':
    main()
