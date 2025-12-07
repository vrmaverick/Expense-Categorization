import numpy as np
from scipy.stats import beta, ks_2samp

def Distributions():
    # Fitted parameters
    a1, b1, loc1, scale1 = 1.53, 17.51, 0.0, 16240.45
    a2, b2, loc2, scale2 = 0.77, 86458.28, 0.0, 10187766.47

    # Reference datasets drawn from the fitted distributions
    data1 = beta(a=a1, b=b1, loc=loc1, scale=scale1).rvs(size=10000)
    data2 = beta(a=a2, b=b2, loc=loc2, scale=scale2).rvs(size=10000)

    return data1,data2

def Hypotesis_testing(sample,data1,data2):
    stat1, p1 = ks_2samp(sample, data1)
    stat2, p2 = ks_2samp(sample, data2)

    print(f"KS vs Dataset 1: stat={stat1:.4f}, p={p1:.4e}")
    print(f"KS vs Dataset 2: stat={stat2:.4f}, p={p2:.4e}")

    # Choose closer distribution by smaller KS statistic (or larger p-value)
    chosen = "Dataset 1" if stat1 < stat2 else "Dataset 2"
    print("Sample is closer to:", chosen)
