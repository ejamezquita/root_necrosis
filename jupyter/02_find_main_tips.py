from glob import glob
import cv2
import numpy as np
from scipy import ndimage, spatial

import os
import argparse
import pandas as pd

from skimage import morphology, graph
import skfmm

import gudhi as gd

genotypes = ['CAL','MLB','222','299','517','521']
imtype = ['Diseased', 'Healthy', 'Binary']
pad = 5

def read_binary_img(filename, threshold=100):
    bimg = cv2.imread(filename)[:,:,0]
    bimg[bimg < threshold] = 0
    bimg[bimg > 0] = 1
    bimg = bimg.astype(bool)

    return bimg

def clean_zeros_2d(img, pad=2):
    foo = np.nonzero(np.any(img, axis=0))[0]
    vceros = np.array([ max([0,foo[0] - pad]), min([img.shape[1], foo[-1]+pad]) ])
    
    foo = np.nonzero(np.any(img, axis=1))[0]
    hceros = np.array([ max([0,foo[0] - pad]), min([img.shape[0], foo[-1]+pad]) ])

    img = img[hceros[0]:hceros[1], vceros[0]:vceros[1]]
    
    return img, vceros, hceros

def pers2numpy(pers):
    bd = np.zeros((len(pers), 3), dtype=float)
    for i in range(len(bd)):
        bd[i, 0] = pers[i][0]
        bd[i, 1:] = pers[i][1]
    return bd
    
def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('runnum', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    runnum = args.runnum
    
    src = '../run{:02d}/'.format(runnum)
    gdst = src + 'gudhi/'
    if not os.path.isdir(gdst):
        os.mkdir(gdst)

    for gidx in range(len(genotypes)):
        
        bfiles = sorted(glob(src + '{}*/*{}*.jpg'.format(imtype[2], genotypes[gidx])))
        print('Total number of files:\t{}'.format(len(bfiles)))

        for idx in range(len(bfiles)):
            
            print(bfiles[idx])
            bimg = read_binary_img(bfiles[idx])
            bname = os.path.splitext(os.path.split(bfiles[idx])[1])[0].split('_ivc')[0].replace(' ','')

            img = bimg.copy()
            img = ndimage.binary_dilation(img, ndimage.generate_binary_structure(2,1), 3)
            img, vceros, hceros = clean_zeros_2d(img)

            label, nums = ndimage.label(img, structure=ndimage.generate_binary_structure(2,1))
            print('Found',nums,'connected components')
            hist, bins = np.histogram(label, bins=range(1,nums+2))
            hargsort = np.argsort(hist)[::-1]

            main = img.copy()
            main[label != bins[hargsort[0]]] = False
            edt = ndimage.distance_transform_edt(~main, return_distances=False, return_indices=True)

            gimg = img.copy()

            rest = img.copy()
            rest[label == bins[hargsort[0]]] = False
            skel = morphology.skeletonize(rest)
            g,nodes = graph.pixel_graph(skel, connectivity=2)

            argleaf = np.nonzero(np.sum(g.A > 0, axis=0) == 1)[0]
            leafx = nodes[argleaf]%skel.shape[1]
            leafy = nodes[argleaf]//skel.shape[1]
            leafz = label[leafy, leafx]

            eidx = np.zeros((len(leafx), 2), dtype=int)
            for i in range(len(eidx)):
                eidx[i] = edt[:, leafy[i], leafx[i]]

            sdist = np.zeros(len(leafx))
            for i in range(len(sdist)):
                sdist[i] = (leafx[i]-eidx[i,1])**2 + (leafy[i]-eidx[i,0])**2
            sdist = np.sqrt(sdist)

            for i in range(nums):
                dmask = leafz == i+1
                if np.sum(dmask) > 0:
                    cidx = np.argmin(sdist[dmask])
                    if sdist[dmask][cidx] < 250:
                        p0 = np.array([leafx[dmask][cidx], leafy[dmask][cidx]])
                        p1 = eidx[dmask][cidx][::-1]
                        
                        lams = np.linspace(0,1, 2*int(sdist[dmask][cidx]))
                        
                        for j in range(len(lams)):
                            line = p0 + lams[j]*(p1 - p0)
                            line = line.astype(int)
                            gimg[ line[1]-pad:line[1]+pad, line[0]-pad:line[0]+pad] = True
                    
                    else:
                        gimg[ label == i+1] = False
                        print(i, sdist[dmask], sep='\t')

            foo, bar = ndimage.label(gimg, structure=ndimage.generate_binary_structure(2,1))
            print('Found',bar,'connected components after processing')

            # # Compute the Geodesic Distance Transform

            m = np.copy(gimg)
            m[0, gimg[0] ] = False
            m = np.ma.masked_array(m, ~gimg)

            gdt = skfmm.distance(m).data

            # # Compute root tips via 0D persistence with geodesic filter

            filename = gdst + bname + '_-_H0_{}_{}_{}_{}.csv'.format(*vceros, *hceros)
            print(filename)

            if not os.path.isfile(filename):
                inv = np.max(gdt) - gdt
                main = np.ravel(inv, 'F')
                cc = gd.CubicalComplex(top_dimensional_cells = inv)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=10)
                cof = cc.cofaces_of_persistence_pairs()
                print(len(cof), len(cof[0]), len(cof[0][0]))
                print(len(cof), len(cof[0]), len(cof[0][1]))
                print(len(cof), len(cof[1]), len(cof[1][0]))

                bd = pers2numpy(pers)
                bd = np.atleast_2d(bd[np.all(bd < np.inf, axis=1), :]).squeeze()
                bd0 = bd[bd[:,0] == 0, 1:]
                bd0 = np.vstack(([0,np.max(gdt)], bd0))
                bd0 = bd0[np.argsort(bd0[:,1])[::-1]]
                lt = bd0[:,1] - bd0[:,0]
                
                foo = np.argsort(main[cof[0][0][:,1]])[::-1]
                rest = np.vstack(([cof[1][0][0], gdt.shape[0]*int(ndimage.center_of_mass(gimg[0])[0])], cof[0][0][foo]))
                tips = np.column_stack((rest[:,0]%gdt.shape[0], rest[:,0]//gdt.shape[0]))
                merge = np.column_stack((rest[:,1]%gdt.shape[0], rest[:,1]//gdt.shape[0]))
                print(np.sum(main[rest] != bd0))
                
                birthdeath = pd.DataFrame(bd0, columns=['birth','death'])
                birthdeath['lifetime'] = lt
                birthdeath = pd.concat((birthdeath, pd.DataFrame(rest, columns=['tipF','endF'])), axis=1)
                birthdeath = pd.concat((birthdeath, pd.DataFrame(tips, columns=['tipX','tipY'])), axis=1)
                birthdeath = pd.concat((birthdeath, pd.DataFrame(merge, columns=['endX','endY'])), axis=1)
                birthdeath.to_csv(filename, index=False)

if __name__ == '__main__':
    main()
