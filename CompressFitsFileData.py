#This code loads in fits files, and only saves the data that is needed for the later imaging analysis. 

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import numpy as np
import math
import os.path
import os
import scipy.optimize
import scipy

D = ["00962", "02978", "03965", "08489", "08490", "10725", "10722", "11846", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']

#Energy band, 
eb = ['eb0.3-1', 'eb1-3', 'eb3-10', 'eb0.8-3', 'alleb']

for m in range(len(eb)):
    for i in range(len(D)):
        #Open the original fits file
        F = get_pkg_data_filename('A'+D[i]+'ds_'+eb[m]+'_0.0625.fits')
        Img = fits.getdata(F, ext=0)

        #number of subpixels with more than 0 counts:
        nbwc = 0
        for u in range(len(Img)):
            for u1 in range(len(Img[0])):
                if Img[u][u1] > 0:
                    nbwc += 1
        #Make a long list of the data from the pixels that do not have 0 counts:
        OIdat, xi, yi = [0]*nbwc, [0]*nbwc, [0]*nbwc
        k = 0
        for u in range(len(Img)):
            for u1 in range(len(Img[0])):
                if Img[u][u1] > 0:
                    OIdat[k] = Img[u][u1]
                    xi[k] = u1
                    yi[k] = u
                    k += 1
        #Save to fits file:
        if os.path.isfile('CF_'+D[i]+'_'+eb[m]+'.fits') == 1:
            os.remove('CF_'+D[i]+'_'+eb[m]+'.fits')
        t = Table([xi, yi, OIdat], names=('xi', 'yi', 'OIdat'))
        t.write('CF_'+D[i]+'_'+eb[m]+'.fits', format='fits')
