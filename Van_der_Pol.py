import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the Van der Pol system
def vdp(t, z, mu):
    x, y = z
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# Parameters
mu = 5.0                 # stiffness level
z0 = [2.0, 0.0]          # initial conditions: x=2, y=0
t_span = [0, 20]         # simulate from t=0 to t=20
t_eval = np.linspace(0, 20, 1000)  # times to record the solution

# Solve using scipy.integrate.solve_ivp
sol = solve_ivp(vdp, t_span, z0, t_eval=t_eval, args=(mu,))

# Extract solution
t = sol.t
x = sol.y[0]
y = sol.y[1]

# Plot the result
plt.plot(t, x, label='x(t)')
plt.plot(t, y, label='y(t)')
plt.title(f'Van der Pol Oscillator (mu = {mu})')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.grid(True)
plt.show()