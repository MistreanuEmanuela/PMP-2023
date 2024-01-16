import arviz as az
import matplotlib.pyplot as plt
import pandas as pd

centered = az.load_arviz_data("centered_eight")
non_centered = az.load_arviz_data("non_centered_eight")


num_chains_centered = centered.posterior.chain.size
print(f"Numarul de lanturi pentru center: {num_chains_centered}")
num_chains_non_centered = non_centered.posterior.chain.size
print(f"Numarul de lanturi pentru non center: {num_chains_non_centered}")

total_samples_centered = centered.posterior.draw.size
print(f"Marimea totala pentru center: {total_samples_centered}")
total_samples_non_centered = non_centered.posterior.draw.size
print(f"Marimea totala pentru center: {total_samples_non_centered}")

az.plot_posterior(centered, hdi_prob=0.95)
plt.title("Distribuție a Posteriori - Model Centrat")
plt.show()

az.plot_posterior(non_centered, hdi_prob=0.95)
plt.title("Distribuție a Posteriori - Model Non-Centrat")
plt.show()

print("----------MU---------")
summaries = pd.concat([az.summary(centered, var_names=['mu']), az.summary(non_centered, var_names=['mu'])])
summaries.index = ['centered', 'non_centered']
print(summaries)

print("-------------TAU-----------")
summaries = pd.concat([az.summary(centered, var_names=['tau']), az.summary(non_centered, var_names=['tau'])])
summaries.index = ['centered', 'non_centered']
print(summaries)

rhats_centered = az.rhat(centered, var_names=['mu', 'tau'])
autocorr_mu_centered = az.autocorr(centered.posterior["mu"].values)
autocorr_tau_centered = az.autocorr(centered.posterior["tau"].values)

rhats_non_centered = az.rhat(non_centered, var_names=['mu', 'tau'])

autocorr_mu_non_centered = az.autocorr(non_centered.posterior["mu"].values)
autocorr_tau_non_centered = az.autocorr(non_centered.posterior["tau"].values)

print("Rezultate pentru modelul centrat:")
print(f"Rhat pentru mu: {rhats_centered['mu'].item()}")
print(f"Rhat pentru tau: {rhats_centered['tau'].item()}")
print(f"Autocorelație pentru mu: {autocorr_mu_centered.mean().item()}")
print(f"Autocorelație pentru tau: {autocorr_tau_centered.mean().item()}")
print()

print("Rezultate pentru modelul necentrat:")
print(f"Rhat pentru mu: {rhats_non_centered['mu'].item()}")
print(f"Rhat pentru tau: {rhats_non_centered['tau'].item()}")
print(f"Autocorelație pentru mu: {autocorr_mu_non_centered.mean().item()}")
print(f"Autocorelație pentru tau: {autocorr_tau_non_centered.mean().item()}")


az.plot_autocorr(centered, var_names=["mu", "tau"], combined=True, figsize=(10, 5))
az.plot_autocorr(non_centered, var_names=["mu", "tau"], combined=True, figsize=(10, 5))
plt.show()

divergences_centered = centered.sample_stats["diverging"].sum()
divergences_non_centered = non_centered.sample_stats["diverging"].sum()

print()
print(f"Numărul de divergențe pentru modelul centrat: {divergences_centered.values}")
print(f"Numărul de divergențe pentru modelul non-centrat: {divergences_non_centered.values}")

az.plot_pair(centered, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model centrat")
plt.show()

az.plot_pair(non_centered, var_names=["mu", "tau"], divergences=True)
plt.suptitle("Model necentrat")
plt.show()