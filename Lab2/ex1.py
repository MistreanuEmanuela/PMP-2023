import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import arviz as az

np.random.seed(1)
#datele pe care le stim

p1 = 0.4 #probabilitate mecanic1
p2 = 0.6 #probabilitate mecanic2

lambda1 = 4 #parametru mecanic1
lambda2 = 6 #parametru mecanic2

#functii de care avem nevoie:
#np.mean--medie
#np.std- deviatie
# distribuitie lambda: stats.expon(0,1/lambda);


valori = np.zeros(10000)
for i in range(10000):
    if np.random.rand() < p1:
        valori[i] = stats.expon(0, 1 / lambda1).rvs()
    else:
        valori[i] = stats.expon(0, 1 / lambda2).rvs()

media = np.mean(valori)
deviatie = np.std(valori)

az.plot_posterior({'x': valori})

print(f"Medie X: {media}")
print(f"Deviatie X: {deviatie}")

plt.show()

