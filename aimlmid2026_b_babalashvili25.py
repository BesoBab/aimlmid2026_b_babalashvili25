import numpy as np
import matplotlib.pyplot as plt

# Data
x = np.array([-9.20, -7.00, -5.00, -3.00, -1.00, 1.00, 3.00, 5.00, 7.00, 9.00])
y = np.array([ 6.00,  5.00,  3.00,  4.00,  1.00, 0.00, -2.00, -3.00, -4.00, -5.00])

# Compute trend line (linear regression)
coefficients = np.polyfit(x, y, 1)
trend_line = np.poly1d(coefficients)

# Plot
plt.figure(figsize=(7,5))
plt.scatter(x, y, color='blue', label='Blue data points')
plt.plot(x, trend_line(x), color='red', linewidth=2, label='Trend line')

plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)

plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Linear Trend Line')
plt.legend()
plt.grid(True)

plt.savefig("correlation_scatter.png", dpi=200)
plt.show()
