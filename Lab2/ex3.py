import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import arviz as az

np.random.seed(1)

pm1_b = 0.5
pm1_s = 0.5
pm2_b = 0.7
pm2_s = 0.3

aruncari = []
for i in range(10):
    if np.random.rand() < pm1_s:
        if np.random.rand() < pm2_s:
            variabila='ss'
        else:
            variabila='sb'
    else:
        variabila='b'
        if np.random.rand() < pm2_s:
            variabila='bs'
        else:
            variabila='bb'
    aruncari.append(''.join(variabila))

print(aruncari)


#scalam pentru 100 de experimente:

experimente = []
exp = np.zeros(1000)
k = 0
exp1=[]
for i in range(100):
    for j in range(10):
        if np.random.rand() < pm1_s:
            if np.random.rand() < pm2_s:
                variabila = 'ss'
                var1=1
            else:
                variabila = 'sb'
                var1=2
        else:
            variabila = 'b'
            if np.random.rand() < pm2_s:
                variabila = 'bs'
                var1=3
            else:
                variabila = 'bb'
                var1=4
        experimente.append(''.join(variabila))
        exp[k] = (var1)
        exp1.append((var1))
        k=k+1;

print(experimente)
ss = experimente.count('ss')
sb = experimente.count('sb')
bs = experimente.count('bs')
bb = experimente.count('bb')
print(exp)
print(ss, sb, bs, bb)
print("Media pentru ss:")
print(np.mean(exp == 1))
print("Media pentru sb:")
print(np.mean(exp == 2))
print("Media pentru bs:")
print(np.mean(exp == 3))
print("Media pentru bb:")
print(np.mean(exp == 4))


az.plot_posterior({'x': exp1})
az.plot_posterior({'x': exp})
plt.show()

