#This code cross-correlates an array of images centered around the aligning sources of each image, with a similar array of images created by merging all other observations. This version of the code smoothes the images before cross-correlating them. 

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
import numpy as np
import math
import os.path
import os
import scipy.optimize
import scipy

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

def counts3sig(sgm):
    #Determines how the Gaussian function distributes counts across different pixels. 
    nc = int(6*sgm+1)
    TC = 0 #Total counts
    for u in range(nc):
        for v in range(nc):
            dfc2 = (u - int(3*sgm))**2 + (v - int(3*sgm))**2
            if dfc2 < (int(3*sgm))**2:
                TC += 1/(sgm*math.sqrt(2*math.pi)) * np.exp(-1*dfc2/(2*sgm**2))
    return TC

def GaussCountsSmooth(Img, sgm):
    #The Img should be a n*m array with integers in all entries.
    #sgm is the value of the 1 sigma used to distribute counts and smooth the image. 
    #This code selects a 3 sigma box around any pixel, and distributes the counts around the circle
    TC = counts3sig(sgm)
    SImg = [[0]* len(Img[0]) for u in range(len(Img))]
    for u in range(len(Img)):
        for v in range(len(Img[0])):
            if Img[u][v] > 0:
                #select the x, y range to select
                u1r = [u-int(3*sgm), u+int(3*sgm)+1]
                if u1r[0] < 0:
                    u1r[0] = 0
                if u1r[1] > len(Img):
                    u1r[1] = len(Img)
                v1r = [v-int(3*sgm), v+int(3*sgm)+1]
                if v1r[0] < 0:
                    v1r[0] = 0
                if v1r[1] > len(Img[0]):
                    v1r[1] = len(Img[0])
                for u1 in range(u1r[0], u1r[1]):
                    for v1 in range(v1r[0], v1r[1]):
                        dfc2 = (u1-u)**2 + (v1-v)**2
                        if dfc2 < (int(3*sgm))**2:
                            SImg[u1][v1] += 1/(sgm*math.sqrt(2*math.pi)*TC) * np.exp(-1*dfc2/(2*sgm**2))
    return SImg

#Smooth the parts of the image relevant for the cross correlation:
SimgAS = [[0]*NAS for i in range(N)]
#Keep track of where the center should be, for the next part.
CIAS = [[[ASx[i]-xOF[u], ASy[i]-yOF[u]] for i in range(NAS)] for u in range(N)]
ShiftSASC = [[[0, 0] for j in range(NAS)] for i in range(N)] #how much the image has to be shifted from its 0, 0 location in x and y. 

for u in range(N):
    for i in range(NAS):
        mts, pts = ASy[i]-5*sps-3*Sgm, ASx[i]-5*sps-3*Sgm
        if 10*sps+6*Sgm+1+pts-xOF[u] > len(ImD[u][0]):
            xr = [pts-xOF[u], len(ImD[u][0])]
        else:
            xr = [pts-xOF[u], pts-xOF[u]+10*sps+6*Sgm+1]
        if pts-xOF[u] < 0:
            xr[0] = 0
            ShiftSASC[u][i][0] = xOF[u] - pts
        if 10*sps+6*Sgm+1+mts-yOF[u] > len(ImD[u]):
            yr = [mts-yOF[u], len(ImD[u])]
        else:
            yr = [mts-yOF[u], mts-yOF[u]+10*sps+6*Sgm+1]
        if mts-yOF[u] < 0:
            yr[0] = 0
            ShiftSASC[u][i][1] = yOF[u] - mts
        if xr[1]-xr[0] > 10*sps+6*Sgm+1:
            print("Range is too large in x: ", xr[1]-xr[0], 10*sps+1)
        if yr[1]-yr[0] > 10*sps+6*Sgm+1:
            print("Range is too large in y: ", yr[1]-yr[0], 10*sps+1)
        #Generate the image
        TiZ = [[0]*(10*sps+6*Sgm+1) for w in range(10*sps+6*Sgm+1)]
        for v in range(yr[0], yr[1]):
            v1 = v-yr[0]+ShiftSASC[u][i][1]
            for w in range(xr[0], xr[1]):
                w1 = w-xr[0]+ShiftSASC[u][i][0]
                TiZ[v1][w1] = ImD[u][v][w]
        #Smooth the interval around that
        TSAS = GaussCountsSmooth(TiZ, Sgm)
        #Select the central 10*sps+1 pixels:
        SimgAS[u][i] = [[0]*(10*sps+1) for w in range(10*sps+1)]
        for v in range(10*sps+1):
            for w in range(10*sps+1):
                SimgAS[u][i][v][w] = TSAS[v+3*Sgm][w+3*Sgm]
                
xOF1 = [0]*N
yOF1 = [0]*N

#First, find the center of each of the aligning source, by fitting the counts distribution around each one with a Gaussian, after merging all images in the region around it. 
for i in range(NAS):
    TSImgAS = [[0]*(10*sps+1) for u in range(10*sps+1)]
    for u in range(N):
        #only take instances where ShiftSASC < 3*Sgm+sps:
        if ShiftSASC[u][i][0] < 3*Sgm+sps:
            if ShiftSASC[u][i][1] < 3*Sgm+sps:
                for v in range(10*sps+1):
                    for w in range(10*sps+1):
                        TSImgAS[v][w] += SimgAS[u][i][v][w]
    #Fit the source center
    SCx, SCy = FindGaussPeak1(TSImgAS, 5*sps, 5*sps, 3*sps)
    #update the previous position with the newly fitted ones:
    ASx[i], ASy[i] = round(ASx[i]+SCx-(5*sps)), round(ASy[i]+SCy-(5*sps))

#Now align the images: 
for u in range(N):
    Imtca = [[0]*((10*sps+1)*6) for _ in range((10*sps+1)*6)]
    for i in range(NAS):
        j = int(math.floor(i/6))
        k = i-(6*j)
        xta = int(round(k*(10*sps+1)))
        yta = int(round(j*(10*sps+1)))
        for v in range(N):
            if v == u:
                continue
            if ShiftSASC[v][i][0] < 3*Sgm+sps:
                if ShiftSASC[v][i][1] < 3*Sgm+sps:
                    for m in range(10*sps):
                        for p in range(10*sps):
                            Imtca[m+yta][p+xta] += SimgAS[v][i][m][p]

    #generate the image to align it against:
    ImtO = [[0]*((10*sps+1)*6) for _ in range((10*sps+1)*6)]
    for i in range(NAS):
        if ShiftSASC[u][i][0] < 3*Sgm+sps:
            if ShiftSASC[u][i][1] < 3*Sgm+sps:
                j = int(math.floor(i/6))
                k = i-(6*j)
                xta = int(round(k*(10*sps+1)))
                yta = int(round(j*(10*sps+1)))
                for m in range(10*sps):
                    for p in range(10*sps):
                        ImtO[m+yta][p+xta] = SimgAS[u][i][m][p]

    #Now cross correlate:
    print("Cross correlation for Obs "+D[u])
    #Make sure both image grids have an odd length in both dimensions. 
    for w in range(len(ImtO)):
        ImtO[w].append(0)
        Imtca[w].append(0)
    ImtO.append([0]*len(ImtO[0]))
    Imtca.append([0]*len(Imtca[0]))
    print('Length of images to cross correlate, should be odd: ', len(ImtO), len(ImtO[0]), len(Imtca), len(Imtca[0]))
    corr = signal.correlate2d(ImtO, Imtca, boundary='fill', mode='same')
    print(len(corr), len(corr[0]))
    print("Cross correlation calculated, now analyse")

    #Now determine the peak of the cross correlation:
    GPx, GPy = FindGaussPeak1(corr, round(0.5*len(corr[0])-0.5), round(0.5*len(corr)-0.5), round(5*sps-1))
    #Find at which pixel is the brightest:
    HCC, HCCx, HCCy = 0, 0, 0
    for m in range(round(0.5*len(corr)-0.5)-5*sps, round(0.5*len(corr)-0.5)+5*sps):
        for p in range(round(0.5*len(corr[0])-0.5)-5*sps, round(0.5*len(corr[0])-0.5)+5*sps):
            if corr[m][p] > HCC:
                HCC = corr[m][p]
                HCCy = m
                HCCx = p
    xOF1[u], yOF1[u] = round(xOF[u]-(HCCx-0.5*len(corr[0])+0.5)), round(yOF[u]-(HCCy-0.5*len(corr)+0.5))
    corr = 0 #To reduce memory load

print("\n\nCorrelation completed, after ", w+1, " runs. New x and y offsets are:")
print('xOF = ', xOF1)
print('yOF = ', yOF1)        
