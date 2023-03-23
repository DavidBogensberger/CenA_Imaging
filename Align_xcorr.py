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

sps = 16 #how many subpixels fit into a single pixel along one axis.

#Select the following observations, for which the streak does not intersect the jet:

D = ["00316", "00962", "02978", "03965", "07800", "08489", "08490", "10725", "10722", "11846", "12155", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "19748", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']
D1 = ["316", "962", "2978", "3965", "7800", "8489", "8490", "10725", "10722", "11846", "12155", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "19748", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']

N = len(D)

#Load in the data. Keep the results separate.
ImD = [0]*N

for i in range(N):
    F = get_pkg_data_filename('A'+D[i]+'ds_alleb_TI_0.0625.fits')
    ImD[i] = fits.getdata(F, ext=0)

#Determine offset of images due to boundaries:
#Adjust the initial estimate of the Cen A center to match the boundaries.
CAC = [[200, 150] for i in range(N)]

#These were the initial guesses of the source centers, that were used to generate the images

xc0 =  [4413, 4304, 4049, 4055, 4534, 4225, 3896, 4280, 4088, 4377, 4562, 3850, 4497, 4104, 4340, 4009, 4003, 4193, 4103, 4091, 4094, 4465, 4207, 4248, 4101, 4078, 4081, 4079, 4085, 4089, 4091, 4092, 4091, 4092, 4095, 4097, 4097, 4096]
yc0 =  [3640, 4435, 4094, 4084, 3757, 4179, 4357, 4400, 4080, 4444, 3778, 4209, 4213, 4109, 4392, 4105, 4104, 4273, 4123, 4084, 4080, 3723, 4415, 4428, 4113, 4115, 4101, 4100, 4085, 4080, 4083, 4089, 4079, 4081, 4082, 4079, 4081, 4082]

Ptrx, Ptry = [0]*N, [0]*N
for i in range(N):
    if xc0[i]-200 < 3584:
        Ptrx[i] = 3584+200-xc0[i]
        CAC[i][0] -= Ptrx[i]
    if yc0[i]-150 < 3584:
        Ptry[i] = 3584+150-yc0[i]
        CAC[i][1] -= Ptry[i]
    
#The approximate positions of the knots in the 20th observation, 20794. 
    
ASx20 = [3451, 3632, 3255, 2555, 4514, 4900, 3402, 3272, 3195, 1487, 1268, 1168, 3432, 4614, 4909, 1791, 4062, 4378, 3723]
ASy20 = [2931, 2870, 1770, 2425, 2727, 1879, 763, 279, 151, 1575, 2428, 2925, 3642, 4331, 4072, 4543, 2234, 2351, 1628]
NAS = len(ASx20)

#Now adjust these by the Ptrx, Ptry, to generate the approximate position of each source in each of the images.
ASxi = [[int(ASx20[j] + sps *(Ptrx[20]-Ptrx[i])) for j in range(NAS)] for i in range(N)]
ASyi = [[int(ASy20[j] + sps *(Ptry[20]-Ptry[i])) for j in range(NAS)] for i in range(N)]

#Use the results of the weighted fit as input estimates of the shift:

ADx3, ADy3 = [13, -2, -4, 1, -4, -9, -11, -9, -8, -13, 23, 18, 2, -1, -19, 7, 12, 3, -11, 4, 9, 24, -3, -24, 4, -2, -3, 7, -5, 6, 0, 0, 5, -4, -10, 4, -11, -3], [-19, 8, -5, 6, 6, -2, -7, -8, -6, -4, 28, -4, -4, -7, 13, -11, -13, 4, 12, -1, 0, 15, -1, 2, 0, -3, -5, -1, -2, 3, 2, 1, 5, -6, -6, 2, -10, 8]

for i in range(len(ADx3)):
    if i != 20:
        ADx3[i] -= ADx3[20]
        ADy3[i] -= ADy3[20]

ADx3[20], ADy3[20] = 0, 0

def FitGauss2Din1D(X, A, mux, muy, sigx, sigy, mx, my, c):
    #D is the 1D data
    #When fitting, make sure that the Nx is constant, and at the correct value, equal to Nxy
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
    #Do one fit, for both directions. Fold the data set into one. 
    #First perform test whether Img with iex0, iey0 and dfc0 can work, or whether that region extends beyond the image. 
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
        #For simplicity, first define a narrower image, IM, that contains the subset of Img useful for this. 
        Nxy = 2*dfc0+1
        #Fold it into one dataset.
        Xl = [i for i in range(Nxy*Nxy)]
        IM = [0]*(Nxy*Nxy)
        a = 0
        for i in range(Nxy):
            for j in range(Nxy):
                IM[a] = Img[iey0-dfc0+i][iex0-dfc0+j]
                a += 1
        #Now fit this with the function FitGauss2Din1D
        f1, f1co = scipy.optimize.curve_fit(FitGauss2Din1D, Xl, IM, p0=[1, dfc0, dfc0, 10, 10, 0, 0, 0.001], bounds=([0.000000001, dfc0*(1-ffc), dfc0*(1-ffc), 0.1, 0.1, -1, -1, -1], [1000, dfc0*(1+ffc), dfc0*(1+ffc), dfc0, dfc0, 1, 1, 10]))
        Perr=np.sqrt(np.diag(f1co))
    else:
        f1, Perr = [-1], -1
        print("Source is too close to corner! ", iex0, iey0, dfc0, len(Img))
    return f1, Perr
        
def FindGaussPeak(Img0, iex, iey, dfc):
    #Finds the Peak of the Gaussian, by iteratively running FindGaussPeakImg
    F1, F1e = FindGaussPeakImg(Img0, iex, iey, dfc, 0.75)
    #print("Best fit 1: ", F1, F1e)
    if len(F1) != 1:
        F2, F2e = FindGaussPeakImg(Img0, round(F1[1]+iex-dfc), round(F1[2]+iey-dfc), dfc, 0.5)
        #print("Best fit 2: ", F2, F2e)
        if len(F2) != 1:
            F3, F3e = FindGaussPeakImg(Img0, round(F2[1]+F1[1]+iex-2*dfc), round(F2[2]+F1[2]+iey-2*dfc), dfc, 0.2)
            #print("Best fit 3: ", F3, F3e)
            #Do a fourth one:
            if len(F3) != 1:
                F4, F4e = FindGaussPeakImg(Img0, round(F3[1]+F2[1]+F1[1]+iex-3*dfc), round(F3[2]+F2[2]+F1[2]+iey-3*dfc), dfc, 0.2)
                #print("Best fit 4: ", F4, F4e)
                Dx4, Dy4 = F3[1]+F2[1]+F1[1]+iex-4*dfc, F3[2]+F2[2]+F1[2]+iey-4*dfc
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

#Now run the cross correlation!    

for u in range(2):
    print("Repetition ", u)
    #Start off by generating the image to compare against.
    #Select the 4 pixel grid around each source. Arrange them in a grid. As there are 19 sources, arrange them in a 4*5.
    Imtca = [[0]*((8*sps+1)*5) for w in range((8*sps+1)*4)]
    for i in range(NAS):
        j = int(math.floor(i/5))
        k = i-(5*j)
        #print(i, j, k)
        xta = int(round(k*(8*sps+1)))
        yta = int(round(j*(8*sps+1)))
        ysi = int(round(ASyi[20][i]-ADy3[20]-4*sps))
        xsi = int(round(ASxi[20][i]-ADx3[20]-4*sps))
        yr1, xr1 = 8*sps+1, 8*sps+1
        if yr1 + ysi > len(ImD[20]):
            if yr1 > len(ImD[20]):
                yr1 = len(ImD[20]) - ysi
            else:
                yr1 = 0
        if xr1 + xsi > len(ImD[20][0]):
            if xr1 > len(ImD[20][0]):
                xr1 = len(ImD[20][0]) - xsi
            else:
                xr1 = 0
        for m in range(yr1):
            for p in range(xr1):
                Imtca[m+yta][p+xta] = ImD[20][m+ysi][p+xsi]
    print("Image to compare against was created!")

    #Now generate the equivalent image for the other observations, to determine the shift.
    #For source 20, that shift is just 0, so that does not need to be done.
    for v in range(N):
        ImtO = [[0]*((8*sps+1)*5) for w in range((8*sps+1)*4)]
        for i in range(NAS):
            j = int(math.floor(i/5))
            k = i-(5*j)
            xta = int(round(k*(8*sps+1)))
            yta = int(round(j*(8*sps+1)))
            ysi = int(round(ASyi[v][i]+ADy3[v]-4*sps))
            xsi = int(round(ASxi[v][i]+ADx3[v]-4*sps))
            print("Compared image, ", v, ", ASxi, ASyi: ", ASxi[v][i], ASyi[v][i])
            yr1, xr1 = 8*sps+1, 8*sps+1
            if yr1 + ysi > len(ImD[v]):
                if yr1 > len(ImD[v]):
                    yr1 = len(ImD[v]) - ysi
                else:
                    yr1 = 0
            if xr1 + xsi > len(ImD[v][0]):
                if xr1 > len(ImD[v][0]):
                    xr1 = len(ImD[v][0]) - xsi
                else:
                    xr1 = 0
            for m in range(yr1):
                for p in range(xr1):
                    ImtO[m+yta][p+xta] = ImD[v][m+ysi][p+xsi]
            
            #Now do the cross correlation!
            print("Cross correlation for Obs " + D[v])
            corr = signal.correlate2d(ImtO, Imtca, boundary='fill', mode='same')

            #Find the peak in the xcorr function, by fitting with a Gaussian
            Gfxc, Gfxce, Dx1, Dy1 = FindGaussPeak(corr, int(0.5*len(Imtca[0])), int(0.5*len(Imtca)), 80)
            print("\nBest fit of source 1 Amplitude: ", round(Gfxc[0], 3), " +- ", round(Gfxce[0], 3))
            print("Best fit of source 1 centre in x: ", round(Gfxc[1] + Dx1, 3), " +- ", round(Gfxce[1], 3))
            print("Best fit of source 1 centre in y: ", round(Gfxc[2] + Dy1, 3), " +- ", round(Gfxce[2], 3))
            print("Best fit of source 1 sigma in x: ", round(Gfxc[3], 3), " +- ", round(Gfxce[3], 3))
            print("Best fit of source 1 sigma in y: ", round(Gfxc[4], 3), " +- ", round(Gfxce[4], 3))
            print("Best fit of source 1 Background x gradient: ", round(Gfxc[5], 5), " +- ", round(Gfxce[5], 5))
            print("Best fit of source 1 Background y gradient: ", round(Gfxc[6], 5), " +- ", round(Gfxce[6], 5))
            print("Best fit of source 1 Background constant: ", round(Gfxc[7], 5), " +- ", round(Gfxce[7], 5))


            print(round(Gfxc[1]), round(Dx1), round(ADx3[v]), len(Imtca[0]), round(Gfxc[2]), round(Dy1), round(ADy3[v]), len(Imtca))
            print("\nThe estimated shift of image ", D[v], ", relative to image ", D[20], " in x is:", round(Gfxc[1] + Dx1 + ADx3[v] - 0.5*len(Imtca[0]) + 1, 3), " +- ", round(Gfxce[1], 3), ", compared to previous estimate: ", ADx3[v])
            print("The estimated shift of image ", D[v], ", relative to image", D[20], " in y is:", round(Gfxc[2] + Dy1 + ADy3[v] - 0.5*len(Imtca) + 1, 3), ", compared to previous estimate: ", ADy3[v])
            ADx3[v] = Gfxc[1] + Dx1 + ADx3[v] - 0.5*len(Imtca[0]) + 1
            ADy3[v] = Gfxc[2] + Dy1 + ADy3[v] - 0.5*len(Imtca) + 1

stsbx = round(sum(ADx3) / N)
stsby = round(sum(ADy3) / N)
for i in range(N):
    ADx3[i] -= stsbx
    ADy3[i] -= stsby
    ADx3[i] = round(ADx3[i])
    ADy3[i] = round(ADy3[i])
    
print('\nShift of images, cross correlation:')
print("ADx3 = ", ADx3)
print("ADy3 = ", ADy3)
