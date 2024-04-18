from glob import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage

import os
import pandas as pd

from skimage import morphology, graph
import argparse

genotypes = ['CAL','MLB','222','299','517','521']
imtype = ['Diseased', 'Healthy', 'Binary']
fs = 16; s = 30
maxdist = 50

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
    
def merge_strands(gimg, maxdist=100, iterations=2):
    main = gimg.copy()
    rest = gimg.copy()
    lines = np.zeros_like(main)
                          
    label, nums = ndimage.label(gimg, structure=ndimage.generate_binary_structure(2,1))
    hist, bins = np.histogram(label, bins=range(1,nums+2))
    hargsort = np.argsort(hist)[::-1]

    print(np.round(100*hist[hargsort[:10]]/np.sum(hist),1))

    # Define the principal connected component and the rest of separate strands

    main[label != bins[hargsort[0]]] = False
    rest[label == bins[hargsort[0]]] = False

    # Only consider the pixels at each end of the strands

    skel = morphology.skeletonize(rest)
    g,nodes = graph.pixel_graph(skel, connectivity=2)

    argleaf = np.nonzero(np.sum(g.A > 0, axis=0) == 1)[0]
    leafx = nodes[argleaf]%skel.shape[1]
    leafy = nodes[argleaf]//skel.shape[1]
    leafz = label[leafy, leafx]

    # Find the pixels of the main component that are closest to these ends

    edt = ndimage.distance_transform_edt(~main, return_distances=False, return_indices=True)
    eidx = edt[:, leafy, leafx]
    sdist = np.sqrt((leafy - eidx[0])**2 + (leafx - eidx[1])**2)
    eidx = eidx.T
    # For each strand, consider all the ends and draw the shortest possible connecting line

    for i in np.unique(leafz):
        dmask = leafz == i
        cidx = np.argmin(sdist[dmask])
        if sdist[dmask][cidx] < maxdist:
            p0 = np.array([leafx[dmask][cidx], leafy[dmask][cidx]])
            p1 = eidx[dmask][cidx][::-1]
            
            lams = np.linspace(0,1, 2*int(sdist[dmask][cidx]))
            line = (p0.reshape(-1,1) + lams*(p1 - p0).reshape(-1,1)).astype(int)
            lines[line[1], line[0]] = True

    lines = ndimage.binary_dilation(lines, ndimage.generate_binary_structure(2,2), iterations)
    gimg = gimg | lines

    foo, bar = ndimage.label(gimg, structure=ndimage.generate_binary_structure(2,1))
    print('Found',bar,'connected components after processing')

    return gimg, bar

def main():
    
    parser = argparse.ArgumentParser(description='Extract a color matrix from a plate')
    parser.add_argument('runnum', metavar='raw_dir', type=int, help='directory where raw images are located')
    args = parser.parse_args()

    runnum = args.runnum
    
    src = '../run{:02d}/'.format(runnum)
    gdst = src + 'diagnostic/'
    pdst = src + 'processed/'
    if not os.path.isdir(gdst):
        os.mkdir(gdst)

    if not os.path.isdir(pdst):
        os.mkdir(pdst)

    for gidx in range(len(genotypes)):
        bfiles = sorted(glob(src + '{}*/*{}*.jpg'.format(imtype[2], genotypes[gidx])))
        print('Total number of files:\t{}'.format(len(bfiles)))
        
        for idx in range(len(bfiles)):
            print(bfiles[idx])
            bname = os.path.splitext(os.path.split(bfiles[idx])[1])[0].split('_ivc')[0].replace(' ','')
            filenames = glob(pdst + bname + '_-_completed_binary*.npy')
            
            if len(filenames) == 0:
            
                bimg = read_binary_img(bfiles[idx])
                img = bimg.copy()
                img, vceros, hceros = clean_zeros_2d(img)

                # # Connect missing bits to the main component

                gimg = img.copy()
                nums1, nums2 = 2,3

                for i in range(6):
                    dist = maxdist * (i+1)
                    while (nums1 > 1) and (nums1 != nums2):
                        nums2 = nums1
                        gimg, nums1 = merge_strands(gimg, dist)
                    nums2 += 1

                if nums1 > 1:
                    label, nums = ndimage.label(gimg, structure=ndimage.generate_binary_structure(2,1))
                    hist, bins = np.histogram(label, bins=range(1,nums+2))
                    hargsort = np.argsort(hist)[::-1]
                    gimg[label != bins[hargsort[0]]] = False

                    foo, bar = ndimage.label(gimg, structure=ndimage.generate_binary_structure(2,1))
                    print('Found',bar,'connected components after ultimate processing')

                filename = pdst + bname + '_-_completed_binary_{}_{}_{}_{}.npy'.format(*vceros, *hceros)
                print(filename)
                np.save(filename, gimg, allow_pickle=True)

                fig, ax = plt.subplots(1,2, figsize=(7,7), sharex=True, sharey=True)
                ax = np.atleast_1d(ax).ravel()

                for i,im in enumerate([img, gimg]):
                    ax[i].imshow(im, cmap='inferno', vmin=0, origin='upper')
                    ax[i].tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
                    
                i = 0
                ax[i].set_title('Original image', fontsize=fs); i+=1
                ax[i].set_title('All components merged', fontsize=fs); i+=1

                fig.tight_layout();

                filename = gdst + 'fixed_strands_-_' + bname
                print(filename)
                plt.savefig(filename+'.png', format='png', bbox_inches='tight', dpi=200)
                plt.close()
        
            else:
                print('Exisiting', filenames)

if __name__ == '__main__':
    main()
