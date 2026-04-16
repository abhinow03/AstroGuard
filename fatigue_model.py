import numpy as np
import matplotlib.pyplot as plt

# simulation parameters
hours = 48
time = np.arange(hours)

fatigue = np.zeros(hours)

# fatigue parameters
workload_rate = 0.08
recovery_rate = 0.05

# events
eva_hours = [10,11,12,13]  # EVA workload
sleep_hours = list(range(22,30))  # rest period

for t in range(1, hours):

    fatigue[t] = fatigue[t-1]

    if t in eva_hours:
        fatigue[t] += workload_rate

    if t in sleep_hours:
        fatigue[t] -= recovery_rate

    fatigue[t] = np.clip(fatigue[t], 0, 1)

plt.figure(figsize=(10,5))
plt.plot(time, fatigue, label="Fatigue Index")
plt.axhline(0.8, linestyle="--", color="red", label="Risk Threshold")

for h in eva_hours:
    plt.axvline(h, color="orange", alpha=0.3)

for h in sleep_hours:
    plt.axvline(h, color="blue", alpha=0.1)

plt.xlabel("Mission Time (hours)")
plt.ylabel("Fatigue Level")
plt.title("Astronaut Fatigue Accumulation and Recovery")
plt.legend()
plt.show()