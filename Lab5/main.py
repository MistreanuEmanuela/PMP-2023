
import pymc3 as pm
import csv
import numpy as np


traffic_data = []
with open("trafic.csv", 'r') as file:
    csvreader = csv.reader(file)
    header = next(csvreader)
    for row in csvreader:
        traffic_data.append(int(row[1]))

intervals = [(4, 7), (7, 8), (8, 16), (16, 19), (19, 24)]
hours_increase = [7, 16]
hours_decrease = [8, 19]

with pm.Model() as model:
    lambda_param = pm.Gamma("lambda", alpha=1, beta=0.1)
    traffic = []
    for i in range(len(traffic_data)):
        interval = None
        for start, end in intervals:
            if start <= i / 60 < end:
                interval = (start, end)
                break

        rate_multiplier = 1.0
        if interval:
            if interval[0] in hours_increase:
                rate_multiplier = 1.2
            elif interval[0] in hours_decrease:
                rate_multiplier = 0.8

        rate = lambda_param * rate_multiplier
        traffic.append(pm.Poisson(name=f"Poisson_{i}", mu=rate))
    # traffic_obs = pm.Poisson(name="traffic_obs", mu=traffic, observed=traffic_data)


with model:
    trace = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

with model:
    posterior_predictive = pm.sample_posterior_predictive(trace, var_names=["traffic_obs"])

interval_endpoints = []

for i in range(len(intervals)):
    interval_start, interval_end = intervals[i]
    poisson_rates = np.mean(posterior_predictive["traffic_obs"][:, i, :], axis=0)
    most_probable_endpoint = interval_start + np.argmax(poisson_rates)
    interval_endpoints.append((interval_start, most_probable_endpoint))

print("Most probable interval endpoints:")
for interval_start, most_probable_endpoint in interval_endpoints:
    print(f"Interval: ({interval_start}, {most_probable_endpoint})")

lambda_values = np.mean(posterior_predictive["traffic_obs"], axis=0)

print("Most probable λ values within intervals:")
for i, (interval_start, most_probable_endpoint) in enumerate(interval_endpoints):
    print(f"Interval: ({interval_start}, {most_probable_endpoint}), λ: {lambda_values[i]:.2f}")

pm.summary(trace, hdi_prob=0.95)
