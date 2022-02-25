import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

def correlate(file1,file2,maskfile='none'):
    
    map1 = hp.read_map(file1)
    map2 = hp.read_map(file2)

    npix = np.shape(map1)[0]
    
    outname   = sys.argv[3]
    
    if maskfile != 'none':
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


def sim_noise_map(map_in: str,map_out: str):
    import random
    rms_map     = hp.read_map(map_in)
    npix        = np.shape(rms_map)[0]
    realization = np.zeros(npix)
    for i in range(npix):
        realization[i] = random.random()*rms_map[i]

    hp.write_map(map_out,realization)

