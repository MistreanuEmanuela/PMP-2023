import numpy as np
from scipy.stats import poisson, norm, expon


# 1. Modelul
lambda_v = 20
plasare_comanda = 2
deviatie_comanda = 0.5
alpha_v = 5

nr_clienti = poisson.rvs(mu=lambda_v)

timp_comanda = norm.rvs(loc=plasare_comanda, scale=deviatie_comanda, size=nr_clienti)

timp_pregatire = expon.rvs(scale=alpha_v, size=nr_clienti)

total_timp = timp_comanda + timp_pregatire

total_timp_h = np.sum(total_timp)

print(f"Numărul de clienți: {nr_clienti}")
print(f"Timpul total de așteptare pentru toți clienții într-o oră: {total_timp_h} minute")

# 2. alpha in 15 min
probabilitate_dorita = 0.95
numar_simulari = 10_000
nr_clienti = poisson.rvs(mu=lambda_v, size= numar_simulari)
timp_comanda = norm.rvs(loc=plasare_comanda, scale=deviatie_comanda, size=numar_simulari)

def timp_servire_sub_15_minute(alpha, numar_simulari, lambda_v, plasare_comanda, deviatie_comanda):
    timp_comanda = norm.rvs(loc=plasare_comanda, scale=deviatie_comanda, size=numar_simulari)
    timp_pregatire = expon.rvs(scale=alpha, size=numar_simulari)
    timp_total = timp_comanda + timp_pregatire
    timp_servire_95 = np.percentile(timp_total, 95)
    return timp_servire_95

alpha_max = 0
alpha_values = np.linspace(10, 0, 1000)
for alpha in alpha_values:
    timp_servire_95 = timp_servire_sub_15_minute(alpha, numar_simulari, lambda_v, plasare_comanda, deviatie_comanda)
    # print(timp_servire_95)
    if timp_servire_95 <= 15:
        alpha_max = alpha
        break

if alpha_max > 0:
    print("Valoarea maximă a lui α pentru care timpul total de servire este sub 15 minute pentru 95% dintre clienți este:", alpha_max)
else:
    print("Nu s-a găsit o valoare a lui α care să îndeplinească condiția.")


#3..
def average_waiting_time(alpha, num_simulations, plasare_comanda, deviatie_comanda):
    waiting_times = []

    for _ in range(num_simulations):
        timp_comanda = norm.rvs(loc=plasare_comanda, scale=deviatie_comanda, size=numar_simulari)
        timp_pregatire = expon.rvs(scale=alpha, size=numar_simulari)
        timp_total = timp_comanda + timp_pregatire
        waiting_time = timp_total
        waiting_times.append(waiting_time)

    average_wait_time = np.mean(waiting_times)
    return average_wait_time


average_wait_time = average_waiting_time(alpha_max, numar_simulari, plasare_comanda, deviatie_comanda)
print(f"Timpul mediu de așteptare pentru a fi servit al unui client cu alpha={alpha_max} este {average_wait_time:.2f} minute.")






