import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import os

print("Starting curve fitting...")
print(f"Working directory: {os.getcwd()}")

data = pd.read_csv('xy_data.csv')
print(f"Loaded {len(data)} data points")

xd = data['x'].values
yd = data['y'].values

n = len(xd)
print(f"Creating {n} t values from 6 to 60")
t_vals = np.linspace(6, 60, n)

def calc_curve(t, theta, M, X):
    th_rad = np.radians(theta)
    x = t * np.cos(th_rad) - np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.sin(th_rad) + X
    y = 42 + t * np.sin(th_rad) + np.exp(M * np.abs(t)) * np.sin(0.3 * t) * np.cos(th_rad)
    return x, y

def loss(params):
    theta, M, X = params
    xp, yp = calc_curve(t_vals, theta, M, X)
    err = np.sum(np.abs(xd - xp) + np.abs(yd - yp))
    return err

bounds = [(0, 50), (-0.05, 0.05), (0, 100)]

print("\nStarting optimization...")
print("This may take 2-3 minutes...")

res = differential_evolution(loss, bounds, seed=42, maxiter=1500, 
                             popsize=25, tol=1e-9, atol=1e-9,
                             workers=1, updating='deferred',
                             disp=True)

theta_f, M_f, X_f = res.x

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"theta = {theta_f:.6f} degrees")
print(f"M = {M_f:.8f}")
print(f"X = {X_f:.6f}")
print(f"\nFinal L1 error: {res.fun:.6f}")
print(f"Average error per point: {res.fun/n:.6f}")
print("="*50)

xf, yf = calc_curve(t_vals, theta_f, M_f, X_f)

err_per_point = np.abs(xd - xf) + np.abs(yd - yf)
print(f"\nError statistics:")
print(f"  Max error: {np.max(err_per_point):.4f}")
print(f"  Median error: {np.median(err_per_point):.4f}")
print(f"  95th percentile: {np.percentile(err_per_point, 95):.4f}")

plt.figure(figsize=(12, 8))
plt.scatter(xd, yd, c='blue', s=8, label='Data Points', alpha=0.5)
plt.plot(xf, yf, 'r-', linewidth=2.5, label='Fitted Curve', alpha=0.9)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.title(f'Parametric Curve Fit (Error: {res.fun:.2f})', fontsize=14)
plt.axis('equal')
plt.tight_layout()
plt.savefig('fit_result.png', dpi=200, bbox_inches='tight')
print("\nPlot saved as 'fit_result.png'")
plt.show()

theta_deg = theta_f
M_val = M_f
X_val = X_f

print("\n" + "="*50)
print("DESMOS FORMAT")
print("="*50)
desmos = f"\\left(t\\cos({theta_deg:.4f})-e^{{{M_val:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\sin({theta_deg:.4f})+{X_val:.4f},42+t\\sin({theta_deg:.4f})+e^{{{M_val:.6f}\\left|t\\right|}}\\cdot\\sin(0.3t)\\cos({theta_deg:.4f})\\right)"
print(desmos)
print("="*50)

print("\nCopy the above line to Desmos with parameter range:")
print("6 < t < 60")