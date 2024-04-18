from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, spatial
import os
import argparse
import pandas as pd
import skfmm
import gudhi as gd

genotypes = ['CAL','MLB','222','299','517','521']
zpad = 10

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

    src = '../run{:02d}/processed/'.format(runnum)

    gdst = '../run{:02d}/gudhi/'.format(runnum)
    ddst = '../run{:02d}/diagnostic/'.format(runnum)
    if not os.path.isdir(gdst):
        os.mkdir(gdst)
    
    for gidx in range(len(genotypes)):

        bfiles = sorted(glob(src + '*{}*.npy'.format(genotypes[gidx])))
        print('Total number of files:\t{}'.format(len(bfiles)))

        for idx in range(len(bfiles)):
        
            bname = os.path.splitext(os.path.split(bfiles[idx])[1])[0].split('_-_')[0]
            print(bname, sep='\n')
            
            filename = gdst + bname + '_-_H0.csv'
            if not os.path.isfile(filename):
            
                gimg = np.load(bfiles[idx], allow_pickle=True)

                # # Compute the Geodesic Distance Transform

                m = np.copy(gimg)
                m[0, gimg[0] ] = False
                m = np.ma.masked_array(m, ~gimg)

                gdt = skfmm.distance(m).data

                fs = 12; fig, ax = plt.subplots(1,3, figsize=(9,5), sharex=True, sharey=True)
                ax = np.atleast_1d(ax).ravel()

                for i,im in enumerate([gimg, gdt, gdt - gimg*np.arange(len(gdt)).reshape(-1,1)]):
                    ax[i].imshow(im, cmap='inferno', origin='upper', vmin=0)
                    ax[i].tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)

                ax[0].set_ylabel(bname, fontsize=fs);  i = 0
                ax[i].set_title('Binarized image', fontsize=fs); i+=1
                ax[i].set_title('Geodesic distance from root base', fontsize=fs); i+=1
                ax[i].set_title('Difference with vertical distance', fontsize=fs); i+=1

                fig.tight_layout()
                filename = ddst + 'geodesic_distance_transform_-_' + bname
                plt.savefig(filename +'.png', format='png', bbox_inches='tight', dpi=200)
                plt.close()

                # # Compute root tips via 0D persistence with geodesic filter
                
                inv = np.max(gdt) - gdt
                main = np.ravel(inv, 'F')
                cc = gd.CubicalComplex(top_dimensional_cells = inv)
                pers = cc.persistence(homology_coeff_field=2, min_persistence=10)
                cof = cc.cofaces_of_persistence_pairs()

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
                
                geodesic = gdt[tips[:,0], tips[:,1]]
                
                filename = gdst + bname + '_-_H0.csv'
                birthdeath = pd.DataFrame(bd0, columns=['birth','death'])
                birthdeath['lifetime'] = lt
                birthdeath = pd.concat((birthdeath, pd.DataFrame(rest, columns=['tipF','endF'])), axis=1)
                birthdeath = pd.concat((birthdeath, pd.DataFrame(tips, columns=['tipX','tipY'])), axis=1)
                birthdeath = pd.concat((birthdeath, pd.DataFrame(merge, columns=['endX','endY'])), axis=1)
                birthdeath['root_geodesic'] = geodesic
                birthdeath.to_csv(filename, index=False)

                # # Must fit 1 out of 3 criteria to be considered a tip
                # 
                # - Long geodesic distance and part of the convex hull
                # - Longer lifespan
                # - Nothing else below a 50px thick strip

                # Convex hull criterion

                tconvexhull = spatial.ConvexHull(np.flip(tips, axis=1))
                thull = tconvexhull.points[tconvexhull.vertices]
                thull = np.vstack((thull, thull[0])).T

                chmask = np.zeros(len(tips), dtype=bool)
                chmask[tconvexhull.vertices] = True
                chmask = chmask & (geodesic > 0.75*gimg.shape[1]) & (lt > 0.1*gimg.shape[1])

                # Lifespan criterion

                lmask = lt > 0.25*gdt.shape[0]
                lmask = lmask | ( (tips[:,0] > np.quantile(tips[:,0], 0.875)) & (lt > max([np.sort(lt)[-10], np.quantile(lt, .975)]) ) )

                # Vertical drop criterion

                zeros = np.zeros(len(tips), dtype=int)

                for i in range(len(zeros)):
                    foo = gimg[:tips[i,0] - 1 , max([tips[i,1] - zpad, 0]):min([tips[i,1]+zpad, gimg.shape[1]])]
                    bar = gimg[tips[i,0] + 1: , max([tips[i,1] - zpad, 0]):min([tips[i,1]+zpad, gimg.shape[1]])]
                    zeros[i] = min([np.sum(foo), np.sum(bar)])

                vmask = (zeros < 25) & (lt > 0.25*gimg.shape[1])

                fs = 15; fig, ax = plt.subplots(1,3, figsize=(9,5), sharex=True, sharey=True)
                ax = np.atleast_1d(ax).ravel()

                for i,mask in enumerate([lmask, chmask, vmask]):
                    ax[i].imshow(gdt, cmap='inferno', origin='upper', vmin=0)
                    ax[i].scatter(tips[mask,1], tips[mask,0], marker='o', color='r', edgecolor='lime', linewidth=1, zorder=3)
                    ax[i].scatter(merge[mask,1], merge[mask,0], marker='D', color='cyan', edgecolor='w', linewidth=1, zorder=3)
                    ax[i].tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)

                ax[0].set_ylabel(bname, fontsize=fs)
                ax[0].axhline(np.quantile(tips[:,0], 0.875), c='yellow', ls='--', lw=1)
                ax[1].plot(*thull, c='yellow', lw=1, ls='--', zorder=2)

                i = 0
                ax[i].set_title('Lifetime criterion', fontsize=fs); i+=1
                ax[i].set_title('Convex hull criterion', fontsize=fs); i+=1
                ax[i].set_title('Vertical drop criterion', fontsize=fs); i+=1

                fig.tight_layout()
                filename = ddst + 'main_root_tip_-_' + bname
                plt.savefig(filename +'.png', format='png', bbox_inches='tight', dpi=200)
                plt.close()
                
                # Save tip mask

                tmask = chmask | vmask | lmask
                filename = gdst + bname + '_-_root_tips.csv'
                print(filename)
                np.savetxt(filename, np.nonzero(tmask)[0].reshape(1,-1), delimiter=',', fmt='%d')

                # # Make a proper geodesic watershed

                watershed = np.zeros(inv.shape, dtype=np.uint8)
                colorval = 1
                colordict = dict()

                for ix in range(len(rest[tmask])-1, -1, -1):
                    print('Iteration:\t', ix, '\n')
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
                    minorlabels = np.setdiff1d(range(nums), mainlabels)
                    print('Main labels:\t',mainlabels,'\nMinor labels:\t',minorlabels,'\n--\n')
                    
                    foo = timg*inv
                    ends = np.asarray(ndimage.maximum_position(foo, label, index=range(1,nums+1)))
                    sdist = spatial.distance_matrix(ends[mainlabels], ends[minorlabels])

                    for i in range(len(mainlabels)):
                        mask = (watershed == 0) & (label == mainlabels[i] + 1)
                        watershed[mask] = colorval
                        extras = np.nonzero(mainlabels[np.argmin(sdist, axis=0)] == mainlabels[i])[0]
                        for j in minorlabels[extras]:
                            mask = (watershed == 0) & (label == j+1)
                            watershed[mask] = colorval
                        
                        colorlist = []
                        for j in range(len(tips[tmask])):
                            if label[tuple(tips[tmask][j])] == mainlabels[i] + 1:
                                colorlist.append(rest[tmask][j,0])
                        colordict[colorval] = colorlist
                        colorval += 1

                fs = 15; cmap = plt.get_cmap('Blues', len(colordict) + 1)
                fig, ax = plt.subplots(1,1, figsize=(6,6), sharex=True, sharey=True)
                ax = np.atleast_1d(ax).ravel()

                a = ax[0].imshow(watershed, cmap=cmap, origin='upper', vmin=-.5, vmax=len(colordict)+.5)
                ax[0].scatter(tips[tmask,1], tips[tmask,0], marker='o', color='r', edgecolor='lime', linewidth=1)
                ax[0].scatter(merge[tmask,1], merge[tmask,0], marker='*', color='white', edgecolor='magenta', linewidth=1)
                ax[0].tick_params(labelleft=False, left=False, bottom=False, labelbottom=False)
                ax[0].set_ylabel(bname, fontsize=fs)
                ax[0].set_title('Root tip-based watershed', fontsize=fs)
                cax = plt.colorbar(a, ticks=range(len(colordict) + 1))

                fig.tight_layout()

                filename = ddst + 'watershed_root_tip_-_' + bname
                plt.savefig(filename +'.png', format='png', bbox_inches='tight', dpi=200)
                plt.close()


if __name__ == '__main__':
    main()
