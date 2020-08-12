import numpy as np
from matplotlib import colors

qpflevs = [.01, .02, .05, .10, .25, .50, .75, 1, 2, 3, 5, 8, 12]

qpfticks = qpflevs#[str(x) for x in qpflevs]

qpfcolors = ['#e3e3e3','#c3c3c3','#898989','#84a281','#77c37d',
             '#349c51','#006d2c','#c8df00','#ffff00','#ffd100',
             '#ffa100','#ff6c00','#ff0000']

qpflevloc = qpflevs
qpfcmap = colors.ListedColormap(qpfcolors[:-1], name='qpfmap')
qpfcmap.set_over(qpfcolors[-1])
qpfnorml = colors.BoundaryNorm(qpflevs, len(qpfcolors)-1, clip=False)

# SNOW (inches)
# Ticks to break the color at
snowlevs = [.5, 1, 2, 4, 8, 12, 16, 20, 24, 36, 48]

# Which values to actually label (can differ from levs)
snowticks = [str(x) for x in snowlevs]

snowcolors = ['#d3d3d3','#a9a9a9','#abd9e9','#74add1','#4575b4',
                '#313695','#e2c5df','#a2529f','#5f0f5f','#380838','black']

snowlevloc = snowlevs
snowcmap = colors.ListedColormap(snowcolors[:-1], name='snowmap')
snowcmap.set_over(snowcolors[-1])
snownorml = colors.BoundaryNorm(snowlevs, len(snowcolors)-1)

# Probability
problevs = np.arange(-5, 105.1, 10)

probticks = ['%d'%p for p in np.arange(0, 100.1, 10)] + ['']

probcolors = ['#a50026','#d73027','#f46d43','#fdae61','#313695','#4575b4',
'#74add1','#abd9e9','#A9A9A9','#D3D3D3','#ffffff'][::-1] # Diverging WhBuRd

problevloc = problevs+5
probcmap = colors.ListedColormap(probcolors, name='probmap')
probnorml = colors.BoundaryNorm(problevs, len(probcolors))

#SLR
slrlevs = [2.5, 5, 7.5, 10, 15, 20, 25, 30]
slrlevloc = slrlevs
slrticks =  slrlevs

slrcolors = ['#ffffcc','#c7e9b4','#7fcdbb','#41b6c4','#1d91c0','#225ea8','#0c2c84']

slrcmap = colors.ListedColormap(slrcolors, name='slrmap')
slrnorml = colors.BoundaryNorm(slrlevs, len(slrcolors))

# WBZH
wbzlevs = [100, 200, 400, 800, 1200, 1600, 2000, 4000]
wbzcolors = ['#fff7bc','#fee391','#fec44f','#fe9929','#ec7014','#cc4c02','#993404']

wbzlevloc = wbzlevs
wbzticks = wbzlevs
wbzcmap = colors.ListedColormap(wbzcolors, name='wbzmap')
wbznorml = colors.BoundaryNorm(wbzlevs, len(wbzcolors))

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    a=np.outer(np.arange(0,1,0.01),np.ones(10)).T

    fig, ax = plt.subplots(3, 1, facecolor='w', figsize=(10, 6))

    ax[0].imshow(a, aspect='auto', cmap=qpfcmap, origin="lower")
    ax[0].set_title('QPF')

    ax[1].imshow(a, aspect='auto', cmap=snowcmap, origin="lower")
    ax[1].set_title('SNOW')

    ax[2].imshow(a, aspect='auto', cmap=probcmap, origin="lower")
    ax[2].set_title('PROBABILITY')

    plt.show()
