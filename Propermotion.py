#This code calculates the proper motion of a selected source in Cen A images, by fitting a 2D Gaussian function across the distribution of counts in each of the aligned images, assuming that the center shifts linearly in x and y coordinates as a function of time. The proper motion is measured along two axes, which are rotated by an angle theta relative to x and y. The 2D gaussian fitting function can have a different extent in the two axes along which motion is measured. This function assumes a steady background around the source, and a consistant shape of the source throughout the observations. The results of the fits are printed out at the end, and are also saved to a file. The calculation of the apparent proper motion is performed in units of subpixels/year. This needs to be converted to other units for use.

from astropy.io import fits
from astropy.table import Table
import numpy as np
import math
import os.path
import os
import scipy.optimize
import scipy

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')
plt.rcParams.update({'font.size': 17})

sps = 16 #The fraction of 1 pixel that is used.
rb = 4 #rebin factor for the fitting

#The list of observations:

D = ["00962", "02978", "03965", "08489", "08490", "10725", "10722", "11846", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']

N = len(D)

#The alignment of images, these numbers describe the offsets from the initial estimates.
xOF =  [4, 4, -1, 9, 11, 10, 7, 16, -21, 0, -1, 21, -8, -13, -3, 10, -5, -9, 4, 22, -7, -3, 3, -5, 2, -6, -1, 5, -8, 4, 10, -1, 7, 5]
yOF =  [-7, 6, -5, 3, 10, 10, 8, 3, 5, 0, 6, -13, 11, 13, -3, -11, 3, 2, 2, -2, -1, 4, 8, -1, 0, 0, 0, -5, -6, 7, -6, -5, 5, -2]

iex0, iey0 = 2446, 2890 #The initial estimate of the central position
dfc = 13 # The number of rebinned subpixels to include on either side of the central position for the fitting.
Nam = 'JKAX4' #Give a name to this particular source. Used for saving the output.
eb = 'alleb' #The energy band to use. 

theta = np.arctan((3200-iex0)/(iey0-2400))

#Load in the data:
Ts = [0]*N #List of times of the observations. 
ImD = [[[0]*(350*sps) for j in range(300*sps)] for i in range(N)] #Imaging data

for i in range(N):
    #Open the compressed file
    F = fits.open('CF_'+D[i]+'_'+eb+'.fits', format='fits')
    xi = F[1].data['xi']
    yi = F[1].data['yi']
    OIdat = F[1].data['OIdat']
    F.close()
    for j in range(len(OIdat)):
        ImD[i][yi[j]][xi[j]] = OIdat[j]
    xi, yi, OIdat = 0, 0, 0
    F = fits.open('A'+D[i]+'ds_'+eb+'_0.0625.fits')
    ts = F[1].data["START"]
    F.close()
    Ts[i] = 50814 + (ts[0]/(24*3600))

def GaussVelFunc(X, v, vp, A, mux, muy, sigx, sigy, c):
    #Fits the data to determine the proper motion of a particular structure described by a distribution of counts, assuming it to be approximately described by a 2D Gaussian function.
    #v measures the motion along the jet axis, defined at an angle theta to the vertical.
    #vp measures the perpendicular proper motion to that, in a direction of -theta. At theta=0, v, and vp correspond to y, and x axes.
    #Assumes T is defined outside of this fitting function.
    #Assumes Nt is previously defined. Nt = len(T)
    #Assumes Nxy is previously defined. This is the length and height of the image within which the fit is performed.
    #sumC needs to be defined, as an array with each element describing the number of counts in the image around a particular jet knot, in a particular observation.
    #This function assumes a constant background
    CFtyx = [0]*(Nt*Nxy*Nxy) #The function to fit to the data
    ijk = 0
    for i in range(Nt):
        for j in range(Nxy):
            for k in range(Nxy):
                lyp = (j-muy)*np.cos(theta) - (k-mux)*np.sin(theta) - v*T[i]
                lxp = (k-mux)*np.cos(theta) + (j-muy)*np.sin(theta) - vp*T[i]
                CFtyx[ijk] = (A * (np.exp(-1*(lyp**2/(2*sigy**2))-1*(lxp**2/(2*sigx**2))) / (2*math.pi*sigx*sigy)) + c)*sumC[i]
                ijk += 1
    return CFtyx

T0 = [(Ts[i]-0.5*(Ts[0]+Ts[-1])) / 365.2422 for i in range(N)] # make sure the T list is in units of years, and is centered on 0. 

#Select the obsIDs that can be used:
#Only take observations that contain enough of the selected region to be useful for this:

T, itf = [], []
for i in range(len(T0)):
    if iex0 - (dfc0*rb) - 10 - xOF[i] > 0:
        if iex0 + (dfc0*rb) + 10 - xOF[i] < len(ImD[i][0]):
            if iey0 - (dfc0*rb) - 10 - yOF[i] > 0:
                if iey0 + (dfc0*rb) + 10 - yOF[i] < len(ImD[i]):
                    T.append(T0[i])
                    itf.append(i)

Nt = len(T)

#Start the loop
Vf, Vpf = 0, 0
fscx, fscy = dfc0+0.5, dfc0+0.5 #fitted source center relative to previous one, should be close to dfc
iexr0, ieyr0 = iex0, iey0 #previous source center
fr = [0]*10
nr = 0
Tr = 21 #total number of runs
BFP = [0]*Tr #Best fit parameters
BFPe = [0]*Tr #Errors of best fit parameters

for nr in range(Tr):
    print("\nRun ", nr)
    vy = Vf*np.cos(theta) + Vpf*np.sin(theta)
    vx = Vpf*np.cos(theta) - Vf*np.sin(theta)
    ffc = 0.5
    iex, iey = [round(fscx*rb + iexr0 - ((dfc+0.5)*rb) + vx*rb*T[i]) for i in range(Nt)], [round(fscy*rb + ieyr0 - ((dfc+0.5)*rb) + vy*rb*T[i]) for i in range(Nt)] #New source centers
    iexn, ieyn = round(fscx*rb + iexr0 - ((dfc+0.5)*rb)), round(fscy*rb + ieyr0 - ((dfc+0.5)*rb)) #The iex0 and iey0 for the next run
    
    Nxy = 2*dfc+1
    print('Input center position of gaussian fit for first obs: ', iex[0], iey[0])
    print('Input center position of gaussian fit for last obs: ', iex[-1], iey[-1])
    print('Initial estimate of position: ', Kcx[KN], Kcy[KN])
    print('Distance from first estimate:', ((iex[0]-Kcx[KN])**2 + (iey[0]-Kcy[KN])**2)**0.5, ((iex[-1]-Kcx[KN])**2 + (iey[-1]-Kcy[KN])**2)**0.5)

    #Make one long list:
    Xl = [i for i in range(Nt*Nxy*Nxy)]
    CDtyx = [0]*(Nt*Nxy*Nxy)
    sumC = [0]*Nt
    a = 0
    for i in range(Nt):
        kta = iex[i]-dfc*rb-xOF[itf[i]]
        jta = iey[i]-dfc*rb-yOF[itf[i]]
        for j in range(Nxy):
            for k in range(Nxy):
                for j1 in range(rb):
                    for k1 in range(rb):
                        CDtyx[a] += ImD[itf[i]][j*rb+jta+j1][k*rb+kta+k1]
                        sumC[i] += ImD[itf[i]][j*rb+jta+j1][k*rb+kta+k1]
                a += 1
    print('sumC = ', sumC)
    #Normalise it by the average: 
    ssC = sum(sumC)/len(sumC)
    sumC = [sumC[i]/ssC for i in range(Nt)]

    #Perform the fit:
    fr1, frco = scipy.optimize.curve_fit(GaussVelFuncBasic1, Xl, CDtyx, p0=[0, 0, 1, dfc, dfc, 20/rb, 20/rb, 0.001], bounds=([-5/rb, -5/rb, 0.00001, dfc*(1-ffc), dfc*(1-ffc), 5/rb, 5/rb, -1000], [5/rb, 5/rb, 100000, dfc*(1+ffc), dfc*(1+ffc), dfc, dfc, 1000]))

    Perr=np.sqrt(np.diag(frco))
    print('\nBest fit parameters, fit ', nr, ' fit: ', fr1, Perr)
    BFP[nr] = fr1
    BFPe[nr] = Perr

    Vf += fr1[0] #Update proper motions with new best fit results
    Vpf += fr1[1]

    fscx, fscy = fr1[3], fr1[4] #fitted source center
    iexr0, ieyr0 = iexn, ieyn #update the source center
    fr = [fr1[i] for i in range(len(fr1))]

#Redefine the best fit parameters:
BFP1 = [[BFP[i][j] for i in range(len(BFP))] for j in range(len(BFP[0]))]
v = [BFP[i][0] for i in range(Tr)]
vp = [BFP[i][1] for i in range(Tr)]
A = [BFP[i][2] for i in range(Tr)]
mux = [BFP[i][3] for i in range(Tr)]
muy = [BFP[i][4] for i in range(Tr)]
sigx = [BFP[i][5] for i in range(Tr)]
sigy = [BFP[i][6] for i in range(Tr)]
c = [BFP[i][7] for i in range(Tr)]

ve = [BFPe[i][0] for i in range(Tr)]
vpe = [BFPe[i][1] for i in range(Tr)]
Ae = [BFPe[i][2] for i in range(Tr)]
muxe = [BFPe[i][3] for i in range(Tr)]
muye = [BFPe[i][4] for i in range(Tr)]
sigxe = [BFPe[i][5] for i in range(Tr)]
sigye = [BFPe[i][6] for i in range(Tr)]
ce = [BFPe[i][7] for i in range(Tr)]

#Save the results to a file: 

if os.path.isfile('BFP_propermotion_'+Nam+'_'+eb+'.fits') == 1:
    os.remove('BFP_propermotion_'+Nam+'_'+eb+'.fits')
t = Table([RID, v, ve, vp, vpe, A, Ae, mux, muxe, muy, muye, sigx, sigxe, sigy, sigye, c, ce], names=('RunID', 'v', 've', 'vp', 'vpe', 'A', 'Ae', 'mux', 'muxe', 'muy', 'muye', 'sigx', 'sigxe', 'sigy', 'sigye', 'c', 'ce'))
t.write('BFP_propermotion_'+Nam+'_'+eb+'.fits', format='fits')

#Also print out the results:

print("\nAfter ", nr, " fits:")
print("Best fit of X-ray source ", Nam, " proper motion in jet direction: ", round(Vf, 5), " +- ", round(Perr[0], 5)+', in units of subpixels per year.')
print("Best fit of X-ray source ", Nam, " proper motion in perpendicular direction: ", round(Vpf, 5), " +- ", round(Perr[1], 5))
print("Best fit of X-ray source ", Nam, " Amplitude: ", round(fr1[2]*sumC[0], 6), " +- ", round(Perr[2]*sumC[0], 6))
print("Best fit of X-ray source ", Nam, " Mean x at T=0: ", round(fr1[3] + iex[0] - dfc, 3), " +- ", round(Perr[3], 3))
print("Best fit of X-ray source ", Nam, " Mean y at T=0: ", round(fr1[4] + iey[0] - dfc, 3), " +- ", round(Perr[4], 3))
print("Best fit of X-ray source ", Nam, " SD in perpendicular direction: ", round(fr1[5], 5), " +- ", round(Perr[5], 5))
print("Best fit of X-ray source ", Nam, " SD in jet direction: ", round(fr1[6], 5), " +- ", round(Perr[6], 5))
