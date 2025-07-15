from scipy.stats import norm

media = 100
sigma = 15
n = 36
z = norm.ppf(0.975)
error = z * (sigma / np.sqrt(n))

print(f"IC 95%: [{media - error:.2f}, {media + error:.2f}]")
