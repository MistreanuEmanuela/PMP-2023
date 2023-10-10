import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az

np.random.seed(1)
w1 = 0.25
w2 = 0.25
w3 = 0.3
w4 = 0.2
#probabilitatile;

latenta=stats.expon(0, 1/4).rvs()


timp = np.zeros(10000)
for i in range(10000):
    ct = np.random.rand()
    if ct < w1:
        tmp = stats.gamma.rvs(4, 0, 1/3)
        timp[i] = tmp+latenta
    elif ct < w1+w2:
        tmp = stats.gamma.rvs(4, 0, 1 / 2)
        timp[i] = tmp + latenta
    elif ct < w1+w2+w3:
        tmp = stats.gamma.rvs(5, 0, 1 / 2)
        timp[i] = tmp + latenta
    else:
        tmp = stats.gamma.rvs(5, 0, 1 / 3)
        timp[i] = tmp + latenta

x_mai_mare = np.mean(timp > 3)
print(x_mai_mare)


az.plot_posterior({'x': timp, 'y': timp > 3})
plt.show()

