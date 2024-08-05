#This code prints out the set of CIAO commands to select the region around the nucleus of Cen A for the following imaging analysis. It is based on an initial visual estimate of the center. This is later corrected in aligning codes.

#Selection of observations:
D = ["00962", "02978", "03965", "08489", "08490", "10725", "10722", "11846", "12156", "13303", "15294", "16276", "17890", "17891", "18461", "19747", "19521", "20794", "21698", "22714", '24322', '23823', '24321', '26405', '24319', '24325', '24323', '26453', '24318', '24324', '24320', '24326', '27344', '27345']

#Initial visual estimate of the location of the center of Cen A:
xc0 =  [4304, 4049, 4055, 4225, 3896, 4280, 4088, 4377, 3850, 4497, 4104, 4340, 4009, 4003, 4193, 4103, 4091, 4094, 4207, 4248, 4101, 4078, 4081, 4079, 4085, 4089, 4091, 4092, 4091, 4092, 4095, 4097, 4097, 4096]
yc0 =  [4435, 4094, 4084, 4179, 4357, 4400, 4080, 4444, 4209, 4213, 4109, 4392, 4105, 4104, 4273, 4123, 4084, 4080, 4415, 4428, 4113, 4115, 4101, 4100, 4085, 4080, 4083, 4089, 4079, 4081, 4082, 4079, 4081, 4082]

#Energy ranges to consider:
ebr = ["300:1000", "1000:3000", "3000:10000", "800:3000"]
ebrl = ["0.3-1", "1-3", "3-10", "0.8-3"]

#Define the x and y range:
xr = [[0]*2 for i in range(N)]
yr = [[0]*2 for i in range(N)]

for i in range(N):
    if xc0[i]-200 > 3584:
        xr[i][0] = str(xc0[i]-200)
    else:
        xr[i][0] = str(3584)
    if xc0[i]+150 < 4608:
        xr[i][1] = str(xc0[i]+150)
    else:
        xr[i][1] = str(4608)
    if yc0[i]-150 > 3584:
        yr[i][0] = str(yc0[i]-150)
    else:
        yr[i][0] = str(3584)
    if yc0[i]+150 < 4608:
        yr[i][1] = str(yc0[i]+150)
    else:
        yr[i][1] = str(4608)

for i in range(N):
    for j in range(len(ebr)):
        print("\npunlearn dmcopy")
        print('dmcopy "acis_'+D[i]+'_repro_nostrk_evt2.fits[energy='+ebr[j]+']" A'+D[i]+'ds_'+ebrl[j]+'keV.fits')
        print("punlearn dmcopy")
        print('dmcopy "A'+D[i]+'ds_'+ebrl[j]+'keV.fits[bin x='+xr[i][0]+':'+xr[i][1]+':0.0625,y='+yr[i][0]+':'+yr[i][1]+':0.0625]" A'+D[i]+'ds_eb'+ebrl[j]+'_0.0625.fits')
