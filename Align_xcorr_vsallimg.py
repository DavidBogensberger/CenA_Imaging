#This code cross-correlates an array of images centered around the aligning sources of each image, with a similar array of images created by merging all other observations.

import matplotlib.pyplot as plt
from astropy.visualization import astropy_mpl_style
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import matplotlib
import os.path
import os
import scipy.optimize
import matplotlib.ticker as ticker
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, StrMethodFormatter
from PIL import Image
import scipy
import scipy.optimize
from scipy.signal import convolve
import time
from scipy import signal

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams.update({'font.size': 17})

sps = 16 #The number of subpixels that fit along one side of an original pixel. 

D = ["00962", "02978", "03965", "08489", "08490", "10725", "10722", "11846", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']

N = len(D)

#The initial estimate of the offsets of each observation relative to the preliminary guess. These lists can be updated to iterate the procedure, based on the previous best fit results.

xOF, yOF = [0]*N, [0]*N

#The centers of the sources used for aligning the images:
ASx =  [3441, 3622, 3864, 3245, 2546, 4513, 4890, 3391, 3261, 3183, 4639, 1479, 1257, 1159, 4419, 4603, 4898, 1781, 2408, 3720, 4367, 3057, 3065, 5360, 4248, 3930, 1830, 2487, 3712, 3116, 4727, 3418, 3346, 3459, 4214, 1564]
ASy =  [2929, 2871, 2825, 1770, 2427, 2718, 1880, 765, 281, 154, 70, 1576, 2430, 2926, 4656, 4330, 4072, 4542, 4566, 1627, 2357, 1929, 2078, 299, 440, 420, 2795, 1972, 2613, 3175, 3239, 3647, 4125, 4496, 4563, 4448]

NAS = len(ASx)

#Load in the data, but from the new compressed files.

ImD = [[[0]*(350*sps) for j in range(300*sps)] for i in range(N)]

for i in range(N):
    F = fits.open('CF_'+D[i]+'_alleb_TI.fits', format='fits')
    xi = F[1].data['xi']
    yi = F[1].data['yi']
    OIdat = F[1].data['OIdat']
    F.close()
    for j in range(len(OIdat)):
        ImD[i][yi[j]][xi[j]] = OIdat[j]


def FitGauss2Din1D(X, A, mux, muy, sigx, sigy, mx, my, c):
    #D is the 1D data
    Nxy = int(len(X)**0.5)
    nC = [0]*(Nxy*Nxy) #Number of counts in the grid.
    ij = 0
    for i in range(Nxy):
        for j in range(Nxy):
            nC[ij] = A * np.exp(-1*(i-muy)**2/(2*sigy**2)-1*(j-mux)**2/(2*sigx**2)) + my * i + mx * j + c
            ij += 1
    return nC

def FindGaussPeakImg(Img, iex0, iey0, dfc0, ffc):
    #This code finds the x, and y coordinate of the peak of an assumed Gaussian count distribution in both x and y
    #Uses the Img as input, with initial estimate of center at iex0, iey0
    #dfc0 is the distance from the center, that will be used here. Must be an integer
    #ffc is the fraction of that distance that the center of the Gaussian can deviate from the center of the image
    stf = 0
    if iex0-dfc0 < 0:
        stf = 1
    elif iex0+dfc0 > len(Img[0])-1:
        stf = 1
    elif iey0-dfc0 < 0:
        stf = 1
    elif iey0+dfc0 > len(Img)-1:
        stf = 1
    if stf == 0: 
        Nxy = int(2*dfc0+1)
        #Fold it into one dataset.
        Xl = [i for i in range(Nxy*Nxy)]
        IM = [0]*(Nxy*Nxy)
        a = 0
        for i in range(Nxy):
            for j in range(Nxy):
                IM[a] = Img[iey0-dfc0+i][iex0-dfc0+j]
                a += 1
        #Fit the data
        f1, f1co = scipy.optimize.curve_fit(FitGauss2Din1D, Xl, IM, p0=[1, dfc0, dfc0, 10, 10, 0, 0, 0.001], bounds=([0.000000001, dfc0*(1-ffc), dfc0*(1-ffc), 0.1, 0.1, -1, -1, -1], [1000, dfc0*(1+ffc), dfc0*(1+ffc), dfc0, dfc0, 1, 1, 10]))
        Perr=np.sqrt(np.diag(f1co))
    else:
        f1, Perr = [-1], -1
        print("Source is too close to corner! ", iex0, iey0, dfc0, len(Img))
    return f1, Perr
        
def FindGaussPeak(Img0, iex, iey, dfc):
    #Finds the Peak of the Gaussian, by iteratively running FindGaussPeakImg
    F1, F1e = FindGaussPeakImg(Img0, iex, iey, dfc, 0.75)
    if len(F1) != 1:
        F2, F2e = FindGaussPeakImg(Img0, round(F1[1]+iex-dfc), round(F1[2]+iey-dfc), dfc, 0.5)
        if len(F2) != 1:
            F3, F3e = FindGaussPeakImg(Img0, round(F2[1]+F1[1]+iex-2*dfc), round(F2[2]+F1[2]+iey-2*dfc), dfc, 0.2)
            if len(F3) != 1:
                F4, F4e = FindGaussPeakImg(Img0, round(F3[1]+F2[1]+F1[1]+iex-3*dfc), round(F3[2]+F2[2]+F1[2]+iey-3*dfc), dfc, 0.2)
                Dx4, Dy4 = F4[1]+F3[1]+F2[1]+F1[1]+iex-4*dfc, F4[2]+F3[2]+F2[2]+F1[2]+iey-4*dfc
                #check if the final position is very different from the original estimate. 
                if (Dx4+F4[1]-iex)**2 + (Dy4+F4[2]-iey)**2 > 400:
                    print("Final position disagrees from input position by more than 20. Suggest you check input to function")
                
            else:
                F4, F4e, Dx4, Dy4 = [-1], [-1], -1, -1
        else:
            F4, F4e, Dx4, Dy4 = [-1], [-1], -1, -1
    else:
        F4, F4e, Dx4, Dy4 = [-1], [-1], -1, -1
    return F4, F4e, Dx4, Dy4

def FindGaussPeak1(Img0, iex, iey, dfc):
    #Same as above, but only run two instances, to make code run faster
    F1, F1e = FindGaussPeakImg(Img0, iex, iey, dfc, 0.75)
    if len(F1) != 1:
        F2, F2e = FindGaussPeakImg(Img0, round(F1[1]+iex-dfc), round(F1[2]+iey-dfc), dfc, 0.5)
        #check if the final position is very different from the original estimate.
        if len(F2) != 1:
            Dx2, Dy2 = F2[1]+F1[1]+iex-2*dfc, F2[2]+F1[2]+iey-2*dfc
            print("Final position of Gauss peak differs from initial guess by: ", math.sqrt((Dx2-iex)**2 + (Dy2-iey)**2))
        else:
            Dx2, Dy2 = -1, -1
    else:
        Dx2, Dy2 = -1, -1
    return Dx2, Dy2

#Keep track of the results with:
xOF1 = [0]*N
yOF1 = [0]*N

#Now run the alignment code:
for w in range(1): #Can change this to run several iterations after one another
    #First, find the center of each of the aligning source, by fitting the counts distribution around each one with a Gaussian, after merging all images in the region around it. 
    for i in range(NAS):
        ImgS = [[0]*(16*sps+1) for j in range(16*sps+1)] #Made larger than it needs to be
        for u in range(N):
            mts, pts = ASy[i]-5*sps, ASx[i]-5*sps
            if 10*sps+1+pts-xOF[u] > len(ImD[u][0]):
                xr = [pts-xOF[u], len(ImD[u][0])]
            else:
                xr = [pts-xOF[u], pts-xOF[u]+10*sps+1]
            if pts-xOF[u] < 0:
                xr[0] = 0
            if 10*sps+1+mts-yOF[u] > len(ImD[u]):
                yr = [mts-yOF[u], len(ImD[u])]
            else:
                yr = [mts-yOF[u], mts-yOF[u]+10*sps+1]
            if mts-yOF[u] < 0:
                yr[0] = 0
            if xr[1]-xr[0] > 10*sps+1:
                print("Range is too large in x: ", xr[1]-xr[0], 10*sps+1)
            if yr[1]-yr[0] > 10*sps+1:
                print("Range is too large in y: ", yr[1]-yr[0], 10*sps+1)
            for m in range(yr[0], yr[1]):
                for p in range(xr[0], xr[1]):
                    ImgS[m-mts+yOF[u]][p-pts+xOF[u]] += ImD[u][m][p]
        #Fit the source center
        SCx, SCy = FindGaussPeak1(ImgS, 5*sps, 5*sps, 3*sps)
        ImgS = 0 #To reduce memory load
        #update the previous position with the newly fitted ones:
        if SCx > 0:
            if SCy > 0:
                ASx[i], ASy[i] = round(SCx+pts), round(SCy+mts)

    #Now align the individual images. First, merge all observations except one, and create a grid of images centered on the aligning sources. 
    for u in range(N):
        print("For obs ", D[u])
        Imtca = [[0]*((10*sps+1)*6) for _ in range((10*sps+1)*6)] #The grid of images to cross correlate
        for i in range(NAS):
            j = int(math.floor(i/6))
            k = i-(6*j)
            xta = int(round(k*(10*sps+1)))
            yta = int(round(j*(10*sps+1)))
            for v in range(N):
                if v == u:
                    continue
                mts, pts = ASy[i]-5*sps, ASx[i]-5*sps
                if 10*sps+1+pts-xOF[v] > len(ImD[v][0]):
                    xr = [pts-xOF[v], len(ImD[v][0])]
                else:
                    xr = [pts-xOF[v], pts-xOF[v]+10*sps+1]
                if pts-xOF[v] < 0:
                    xr[0] = 0
                if 10*sps+1+mts-yOF[v] > len(ImD[v]):
                    yr = [mts-yOF[v], len(ImD[v])]
                else:
                    yr = [mts-yOF[v], mts-yOF[v]+10*sps+1]
                if mts-yOF[v] < 0:
                    yr[0] = 0
                if xr[1]-xr[0] > 10*sps+1:
                    print("Range is too large in x: ", xr[1]-xr[0], 10*sps+1)
                if yr[1]-yr[0] > 10*sps+1:
                    print("Range is too large in y: ", yr[1]-yr[0], 10*sps+1)
                for m in range(yr[0], yr[1]):
                    for p in range(xr[0], xr[1]):
                        Imtca[m-mts+yOF[v]+yta][p-pts+xOF[v]+xta] += ImD[v][m][p]
        #Generate the image I will align it against.
        ImtO = [[0]*((10*sps+1)*6) for _ in range((10*sps+1)*6)]
        for i in range(NAS):
            j = int(math.floor(i/6))
            k = i-(6*j)
            xta = int(round(k*(10*sps+1)))
            yta = int(round(j*(10*sps+1)))
            mts, pts = ASy[i]-5*sps, ASx[i]-5*sps
            if 10*sps+1+pts-xOF[u] > len(ImD[u][0]):
                xr = [pts-xOF[u], len(ImD[u][0])]
            else:
                xr = [pts-xOF[u], pts-xOF[u]+10*sps+1]
            if pts-xOF[u] < 0:
                xr[0] = 0
            if 10*sps+1+mts-yOF[u] > len(ImD[u]):
                yr = [mts-yOF[u], len(ImD[u])]
            else:
                yr = [mts-yOF[u], mts-yOF[u]+10*sps+1]
            if mts-yOF[u] < 0:
                yr[0] = 0
            if xr[1]-xr[0] > 10*sps+1:
                print("Range is too large in x: ", xr[1]-xr[0], 10*sps+1)
            if yr[1]-yr[0] > 10*sps+1:
                print("Range is too large in y: ", yr[1]-yr[0], 10*sps+1)
            for m in range(yr[0], yr[1]):
                for p in range(xr[0], xr[1]):
                    ImtO[m-mts+yOF[u]+yta][p-pts+xOF[u]+xta] += ImD[u][m][p]

        print("Cross correlation for Obs " + D[u])
        #Make sure both image grids have an odd length in both dimensions. 
        for w in range(len(ImtO)):
            ImtO[w].append(0)
            Imtca[w].append(0)
        ImtO.append([0]*len(ImtO[0]))
        Imtca.append([0]*len(Imtca[0]))
        corr = signal.correlate2d(ImtO, Imtca, boundary='fill', mode='same')
        print("Cross correlation completed")

        GPx, GPy = FindGaussPeak1(corr, round(0.5*len(corr[0])-0.5), round(0.5*len(corr)-0.5), round(5*sps-1))
        #Find at which pixel the cross correlation is brightest:
        HCC, HCCx, HCCy = 0, 0, 0
        for m in range(round(0.5*len(corr)-0.5)-5*sps, round(0.5*len(corr)-0.5)+5*sps):
            for p in range(round(0.5*len(corr[0])-0.5)-5*sps, round(0.5*len(corr[0])-0.5)+5*sps):
                if corr[m][p] > HCC:
                    HCC = corr[m][p]
                    HCCy = m
                    HCCx = p
        #update the estimate of the offset of image u relative to the other observations:
        xOF1[u], yOF1[u] = round(xOF[u]-(HCCx-0.5*len(corr[0])+0.5)), round(yOF[u]-(HCCy-0.5*len(corr)+0.5))

print("\n\nCorrelation completed, after ", w+1, " runs. New x and y offsets are:")
print('xOF = ', xOF1)
print('yOF = ', yOF1) 
