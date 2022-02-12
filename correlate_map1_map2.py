import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 3:
    raise SystemExit("WRONG! Usage: python [sys.argv[0]] [map1] [map2] [outfile name] [mask (optional)]")

filename1 = sys.argv[1]
filename2 = sys.argv[2]

map1 = hp.read_map(filename1)
map2 = hp.read_map(filename2)

npix = np.shape(map1)[0]

outname   = sys.argv[3]

if len(sys.argv) == 5:
    maskfile = sys.argv[4]

    mask = hp.read_map(maskfile)

    map1masked = []
    map2masked = []
    
    for i in range(npix):
        if mask[i] == hp.UNSEEN or mask[i] == 0.0:
            map1[i] = 0.0
            map2[i] = 0.0
        else:
            map1masked.append(map1[i])
            map2masked.append(map2[i])

    m, b = np.polyfit(map1masked,map2masked,1)

    x = np.linspace(np.min(map1masked),np.max(map1masked),1000)
    
    plt.scatter(map1masked,map2masked,color='k')
    plt.plot(x,m*x+b,color='red')
    plt.show()

            
            
else:    
    m, b = np.polyfit(map1,map2,1)

    x = np.linspace(np.min(map1),np.max(map1),1000)
    
    plt.scatter(map1,map2,color='k')
    plt.plot(x,m*x+b)
    plt.show()
