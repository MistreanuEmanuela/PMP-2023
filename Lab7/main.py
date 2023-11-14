import pandas as pd
import pymc3 as pm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import theano.tensor as tt

df = pd.read_csv('auto-mpg.csv')

sns.scatterplot(x='horsepower', y='mpg', data=df)
plt.title('Relația dintre CP și mpg')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.show()

filtered_df = df[df['horsepower'] != '?']

sns.scatterplot(x='horsepower', y='mpg', data=filtered_df)
plt.title('Relația dintre CP și mpg')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.show()
print(len(filtered_df))

with pm.Model() as model:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10)
    a = np.array(filtered_df['horsepower'])
    mu = alpha + beta * filtered_df['horsepower']
    sigma = pm.HalfNormal('sigma', sd=1)
    mpg = pm.Normal('mpg', mu=mu, sd=sigma, observed=filtered_df['mpg'])

with model:
    map_estimate = pm.find_MAP()
    alpha_map = map_estimate['alpha']
    beta_map = map_estimate['beta']
    print(f'Dreapta de regresie: y = {alpha_map:.2f} + {beta_map:.2f} * CP')


sns.lmplot(x='horsepower', y='mpg', data=filtered_df, ci=None, line_kws={'color': 'red'})
plt.title('Regresia liniară și regiunea de încredere')
plt.xlabel('Cai Putere (CP)')
plt.ylabel('Mile pe Galon (mpg)')
plt.show()
