import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from utils import bass_model, adopters_per_period, years, revenue
import numpy as np





# Provide initial guesses and handle potential optimization warnings
initial_guess = [0.01, 0.1, 150]
try:
    popt, pcov = curve_fit(bass_model, years, revenue, p0=initial_guess, bounds=(0, [1, 1, 200]))
    p, q, M = popt
except Exception as e:
    print(f"Optimization failed: {e}")
    p, q, M = initial_guess  # Default to initial values in case of failure

# Generate future predictions
years_future = np.arange(2023, 2035)
predicted_revenue = bass_model(years_future, p, q, M)

# Estimate number of adopters per period
adopters = adopters_per_period(years_future, p, q, M)

# Plot the diffusion of innovation
plt.figure(figsize=(8, 5))
plt.plot(years, revenue, 'ro', label='Actual Revenue')
plt.plot(years_future, predicted_revenue, 'b-', label='Predicted Diffusion')
plt.xlabel('Year')
plt.ylabel('Market Revenue (Billion $)')
plt.title('Bass Diffusion Model for AI Chip Market')
plt.legend()
plt.grid()
plt.show()

# Plot number of adopters over time
plt.figure(figsize=(8, 5))
plt.bar(years_future, adopters, color='orange', alpha=0.7, label='Adopters per Year')
plt.xlabel('Year')
plt.ylabel('Number of Adopters')
plt.title('Estimated Number of Adopters Per Period')
plt.legend()
plt.grid()
plt.show()

# Display estimated parameters
print(f"Estimated parameters:\nCoefficient of Innovation (p): {p:.4f}\nCoefficient of Imitation (q): {q:.4f}\nMarket Potential (M): {M:.2f}")
