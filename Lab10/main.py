import pymc3 as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az
az.style.use('arviz-darkgrid')

#Ex 1 a:
def main():
    dummy_data = np.loadtxt('dummy.csv')
    x_1 = dummy_data[:, 0]
    y_1 = dummy_data[:, 1]
    order = 5
    x_1p = np.vstack([x_1**i for i in range(1, order+1)])
    x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
    y_1s = (y_1 - y_1.mean()) / y_1.std()
    plt.scatter(x_1s[0], y_1s)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    with pm.Model() as model_l:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + beta * x_1s[0]
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_l = pm.sample(2000, return_inferencedata=True)

    with pm.Model() as model_p:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_1p)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_p = pm.sample(2000, return_inferencedata=True)

    x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    alpha_p_post = idata_p.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_post = idata_p.posterior['beta'].mean(("chain", "draw")).values
    y_p_post = alpha_p_post + np.dot(beta_p_post, np.vstack([x_new**i for i in range(1, order+1)]))
    plt.plot(x_new, y_p_post, 'C3', label=f'model order {order}')

#Ex 1 b:
    with pm.Model() as model_p_100:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_1p)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_p_100 = pm.sample(2000, return_inferencedata=True)

    alpha_p_100_post = idata_p_100.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_100_post = idata_p_100.posterior['beta'].mean(("chain", "draw")).values
    y_p_sd_100_post = alpha_p_100_post + np.dot(beta_p_100_post, np.vstack([x_new**i for i in range(1, order+1)]))
    plt.plot(x_new, y_p_sd_100_post, 'C4', label=f'model order {order}, sd=100')
    plt.legend()

    with pm.Model() as model_p_array:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=100, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_1p)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_p_array = pm.sample(2000, return_inferencedata=True)

    alpha_p_array_post = idata_p_array.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_array_post = idata_p_array.posterior['beta'].mean(("chain", "draw")).values
    y_p_array_post = alpha_p_array_post + np.dot(beta_p_array_post, np.vstack([x_new**i for i in range(1, order+1)]))
    plt.plot(x_new, y_p_array_post, 'C5', label=f'model order {order}, sd=100')
    plt.legend()
    plt.show()


# Exercitiul 2:
    dummy_data_500 = np.loadtxt('dummy.csv')
    x_2 = dummy_data_500[:, 0]
    y_2 = dummy_data_500[:, 1]
    min_x = min(x_2)
    max_x = max(x_2)
    vector_random = np.random.uniform(min_x, max_x, size=500 - len(x_2))
    x_2.extend(vector_random)
    min_y = min(y_2)
    max_y = max(y_2)
    vector_random = np.random.uniform(min_y, max_y, size=500 - len(y_2))
    y_2.extend(vector_random)
    order = 5
    x_2p = np.vstack([x_2 ** i for i in range(1, order + 1)])

    with pm.Model() as model_p_500:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_2p)
        y_pred = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=(y_2 - y_2.mean()) / y_2.std())
        idata_p_500 = pm.sample(2000, return_inferencedata=True)


    x_new_500 = np.linspace(x_2.min(), x_2.max(), 100)
    alpha_p_500_post = idata_p_500.posterior['alpha'].mean(("chain", "draw")).values
    beta_p_500_post = idata_p_500.posterior['beta'].mean(("chain", "draw")).values
    y_p_500_post = alpha_p_500_post + np.dot(beta_p_500_post, np.vstack([x_new_500 ** i for i in range(1, order + 1)]))
    plt.plot(x_new_500, y_p_500_post, 'C6', label=f'model order {order}, 500 data points')
    plt.scatter(x_2, (y_2 - y_2.mean()) / y_2.std(), c='C6', marker='.')
    plt.legend()
    plt.show()

#Exercitiul 3:

    order_cubic = 3
    x_1p_cubic = np.vstack([x_1 ** i for i in range(1, order_cubic + 1)])
    with pm.Model() as model_cubic:
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=order_cubic)
        epsilon = pm.HalfNormal('epsilon', 5)
        miu = alpha + pm.math.dot(beta, x_1p_cubic)
        y_pred_cubic = pm.Normal('y_pred', mu=miu, sigma=epsilon, observed=y_1s)
        idata_cubic = pm.sample(2000, return_inferencedata=True)

    x_new_cubic = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
    alpha_cubic_post = idata_cubic.posterior['alpha'].mean(("chain", "draw")).values
    beta_cubic_post = idata_cubic.posterior['beta'].mean(("chain", "draw")).values
    y_cubic_post = alpha_cubic_post + np.dot(beta_cubic_post, np.vstack([x_new_cubic ** i for i in range(1, order_cubic + 1)]))
    plt.plot(x_new_cubic, y_cubic_post, 'C7', label=f'cubic model')
    plt.legend()

    models = [model_l, model_p, model_cubic]
    idata_models = [idata_l, idata_p, idata_cubic]

    waic_results = [az.waic(idata) for idata in idata_models]
    loo_results = [az.loo(idata) for idata in idata_models]

    for i, (waic, loo, label) in enumerate(zip(waic_results, loo_results, ['Linear', 'Quadratic', 'Cubic'])):
        print(f'{label} Model:')
        print(f'WAIC: {waic.waic.values[0]:.2f}')
        print(f'LOO: {loo.loo.values[0]:.2f}')

    waic_values = [waic.waic.values[0] for waic in waic_results]
    loo_values = [loo.loo.values[0] for loo in loo_results]

    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(len(models)), waic_values, alpha=0.5, label='WAIC')
    plt.bar(np.arange(len(models)), loo_values, alpha=0.5, label='LOO')
    plt.xticks(np.arange(len(models)), ['Linear', 'Quadratic', 'Cubic'])
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()