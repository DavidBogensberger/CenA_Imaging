#This code extracts the best fit results from the output of the Propermotion.py file, and corrects the results for possible offsets found from the proper motions of the aligning sources, and adjusts the errors to include uncertainty originating from the alignment process. It outputs the proper motion in the parallel and perpendicular direction, as well as the total proper motion of each jet knot.

from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits
from astropy.table import Table
import numpy as np
import math
import scipy
import scipy.optimize
import os.path
import os

rb = 4 #The rebin factor.

Nam = 'JKAX4' #The name of the source
eb = 'alleb' #The energy band to use

def invpc(x, A1, c):
    y = A1/x + c
    return y

F = fits.open('BFP_propermotion_'+Nam+'_'+eb+'.fits')
v = F[1].data["v"]
ve = F[1].data["ve"]
vp = F[1].data["vp"]
vpe = F[1].data["vpe"]
a = F[1].data['A']
F.close()

#Convert the proper motions into units of a fraction of the speed of light. 
V = sum(v) * rb*3.8e6*0.492*math.pi*3.26156/(180*60*60*16)
Vp = sum(vp) * rb*3.8e6*0.492*math.pi*3.26156/(180*60*60*16)
Ve = ve[-1] * rb*3.8e6*0.492*math.pi*3.26156/(180*60*60*16)
Vpe = vpe[-1] * rb*3.8e6*0.492*math.pi*3.26156/(180*60*60*16)
A = a[-1]

#Subtract the offset measured in the mean velocity of the brightest jet knots:
xo = -0.06383424997277917
yo = 0.03399838036457726

V -= yo*np.cos(math.pi*theta[i]/180) - xo*np.sin(math.pi*theta[i]/180)
Vp -= yo*np.sin(math.pi*theta[i]/180) + xo*np.cos(math.pi*theta[i]/180)

#Add the error found from the aligning sources to the result:
seta = invpc(A, 3.83375026, 0.07570174)
Ve = (Ve**2+seta**2+0.1**2)**0.5
Vpe = (Vpe[i][m]**2+seta**2+0.1**2)**0.5

#Also determine the total proper motions and errors:

VT = [(V[i]**2 + Vp[i]**2)**0.5 for i in range(NK)]
VTe = [(Ve[i]**2 + Vpe[i]**2)**0.5 for i in range(NK)]

print('\nThe parallel proper motion: ', V, Ve)
print('The perpendicular proper motion: ', Vp, Vpe)
print('The total proper motion: ', VT, VTe)
