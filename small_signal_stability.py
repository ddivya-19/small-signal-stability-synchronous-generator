import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.linalg import eigvals
# Infinite bus
V = 1.0                    # pu
# Synchronous-machine parameters
H = 3.5                    # inertia constant, seconds
D = 1.0                    # damping coefficient, pu
Xd = 1.8                   # d-axis synchronous reactance, pu
Xd_prime = 0.3             # d-axis transient reactance, pu
Tdo_prime = 8.0            # transient open-circuit time constant, s
# Excitation
Efd = 1.2                  # field voltage, pu
# Mechanical input
Pm_initial = 0.8           # initial mechanical input, pu
# System frequency
f = 50.0                   # Hz
omega_b = 2.0 * np.pi * f
# ============================================================
# 2. GENERATOR EQUATIONS
# ============================================================

def electrical_power(delta, Eq_prime):
    """
    Electrical power output.

    Pe = (Eq' * V / Xd') * sin(delta)
    """
  return (Eq_prime * V / Xd_prime) * np.sin(delta)

def d_axis_current(delta, Eq_prime):
    """
    Simplified d-axis current relationship.

    Id = (Eq' - V*cos(delta)) / Xd'
    """
  return (Eq_prime - V * np.cos(delta)) / Xd_prime

def generator_model(t, x, Pm=Pm_initial, D_value=D):
    """
    Nonlinear third-order synchronous-generator model.

    State vector:
        x[0] = delta
        x[1] = Delta omega
        x[2] = Eq'

    Returns:
        dx/dt
    """
    delta = x[0]
    dw = x[1]
    Eq_prime = x[2]

    # Electrical power
    Pe = electrical_power(delta, Eq_prime)

    # d-axis current
    Id = d_axis_current(delta, Eq_prime)

    # --------------------------------------------------------
    # Differential equations
    # --------------------------------------------------------
    # Rotor-angle equation
    ddelta_dt = omega_b * dw
    # Swing equation
    ddw_dt = (
        Pm
        - Pe
        - D_value * dw
    ) / (2.0 * H)

    # Transient EMF equation
    dEq_dt = (
        Efd
        - Eq_prime
        - (Xd - Xd_prime) * Id
    ) / Tdo_prime

    return np.array([
        ddelta_dt,
        ddw_dt,
        dEq_dt
    ])


# ============================================================
# 3. EQUILIBRIUM OPERATING POINT
# ============================================================
def equilibrium_function(x):
    """
    At equilibrium:

        dx/dt = 0
    """
    return generator_model(
        t=0.0,
        x=x,
        Pm=Pm_initial,
        D_value=D
    )
# Initial guess
x_guess = np.array([
    0.5,    # delta, rad
    0.0,    # speed deviation
    1.1     # Eq', pu
])


# Solve equilibrium
equilibrium_solution = root(
    equilibrium_function,
    x_guess
)


if not equilibrium_solution.success:
    raise RuntimeError(
        "Equilibrium solver did not converge."
    )
x0 = equilibrium_solution.x
delta0 = x0[0]
dw0 = x0[1]
Eq0 = x0[2]
print("\n" + "=" * 60)
print("EQUILIBRIUM OPERATING POINT")
print("=" * 60)

print(f"Rotor angle delta0 = {np.rad2deg(delta0):.6f} degrees")
print(f"Speed deviation dw0 = {dw0:.8f} pu")
print(f"Transient EMF Eq0    = {Eq0:.6f} pu")

print("\nEquilibrium residual:")
print(equilibrium_function(x0))
# ============================================================
# 4. NUMERICAL JACOBIAN
# ============================================================
def numerical_jacobian(fun, x, epsilon=1e-6):
    """
    Calculates Jacobian using central finite differences.

    A[i,j] = partial f_i / partial x_j
    """

    x = np.asarray(x, dtype=float)

    n = len(x)

    A = np.zeros((n, n))

    for j in range(n):

        dx = np.zeros(n)
        dx[j] = epsilon

        f_plus = fun(x + dx)
        f_minus = fun(x - dx)
        A[:, j] = (
            f_plus - f_minus
        ) / (2.0 * epsilon)

    return A
# Calculate A matrix
A = numerical_jacobian(
    equilibrium_function,
    x0
)
print("\n" + "=" * 60)
print("LINEARIZED STATE MATRIX A")
print("=" * 60)

print(A)
# ============================================================
# 5. EIGENVALUE ANALYSIS
# ============================================================
eigenvalues = eigvals(A)

print("\n" + "=" * 60)
print("EIGENVALUE ANALYSIS")
print("=" * 60)

for i, lam in enumerate(eigenvalues):

    sigma = np.real(lam)
    omega = np.imag(lam)

    magnitude = np.abs(lam)

    if magnitude > 1e-12:
        damping_ratio = -sigma / magnitude
    else:
        damping_ratio = np.nan

    oscillation_frequency = abs(omega) / (2.0 * np.pi)

    print(f"\nMode {i + 1}")
    print(f"Eigenvalue             = {lam}")
    print(f"Real part              = {sigma:.6f}")
    print(f"Imaginary part         = {omega:.6f}")
    print(f"Damping ratio          = {damping_ratio:.6f}")
    print(
        f"Oscillation frequency  = "
        f"{oscillation_frequency:.6f} Hz"
    )
# Stability decision based on eigenvalues
stable = np.all(np.real(eigenvalues) < 0)

print("\nSmall-signal stability:", end=" ")

if stable:
    print("STABLE")
else:
    print("UNSTABLE")
# ============================================================
# 6. EIGENVALUE PLOT
# ============================================================

plt.figure(figsize=(7, 6))
plt.axhline(
    0,
    linewidth=1
)
plt.axvline(
    0,
    linewidth=1
)
plt.scatter(
    np.real(eigenvalues),
    np.imag(eigenvalues),
    s=80
)

for i, lam in enumerate(eigenvalues):

    plt.annotate(
        f"λ{i+1}",
        (
            np.real(lam),
            np.imag(lam)
        )
    )

plt.xlabel("Real Part")
plt.ylabel("Imaginary Part")
plt.title("Eigenvalue Distribution")

plt.grid(True)
plt.tight_layout()

plt.savefig(
    "eigenvalues.png",
    dpi=300
)

plt.show()
# ============================================================
# 7. DISTURBANCE SIMULATION
# ============================================================

# Small step disturbance in mechanical power
Pm_disturbed = Pm_initial + 0.02

disturbance_time = 1.0
def disturbed_model(t, x):
    """
    Mechanical input changes slightly at t = 1 second.
    """
    if t < disturbance_time:
        Pm = Pm_initial
    else:
        Pm = Pm_disturbed

    return generator_model(
        t,
        x,
        Pm=Pm,
        D_value=D
    )
# Simulation time
t_start = 0.0
t_end = 10.0

t_eval = np.linspace(
    t_start,
    t_end,
    5000
)


solution = solve_ivp(
    disturbed_model,
    (t_start, t_end),
    x0,
    t_eval=t_eval,
    method="RK45",
    rtol=1e-8,
    atol=1e-10
)


if not solution.success:
    raise RuntimeError(
        "Disturbance simulation failed."
    )


# Extract states
time = solution.t

delta = solution.y[0]
dw = solution.y[1]
Eq_prime = solution.y[2]


# ============================================================
# 8. CALCULATE ELECTRICAL POWER
# ============================================================

Pe = electrical_power(
    delta,
    Eq_prime
)
# ============================================================
# 9. ROTOR ANGLE RESPONSE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    np.rad2deg(delta),
    linewidth=1.5
)

plt.axvline(
    disturbance_time,
    linestyle="--",
    label="Disturbance"
)

plt.xlabel("Time (s)")
plt.ylabel("Rotor Angle δ (degrees)")
plt.title("Rotor Angle Response")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "rotor_angle_response.png",
    dpi=300
)

plt.show()
# ============================================================
# 10. SPEED DEVIATION RESPONSE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    dw,
    linewidth=1.5
)

plt.axvline(
    disturbance_time,
    linestyle="--",
    label="Disturbance"
)

plt.xlabel("Time (s)")
plt.ylabel("Speed Deviation Δω (pu)")
plt.title("Speed Deviation Response")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "speed_deviation_response.png",
    dpi=300
)

plt.show()


# ============================================================
# 11. TRANSIENT EMF RESPONSE
# ============================================================

plt.figure(figsize=(9, 5))

plt.plot(
    time,
    Eq_prime,
    linewidth=1.5
)

plt.axvline(
    disturbance_time,
    linestyle="--",
    label="Disturbance"
)
plt.xlabel("Time (s)")
plt.ylabel("Transient EMF E'q (pu)")
plt.title("Transient EMF Response")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "transient_emf_response.png",
    dpi=300
)

plt.show()
# ============================================================
# 12. ELECTRICAL POWER RESPONSE
# ============================================================
plt.figure(figsize=(9, 5))

plt.plot(
    time,
    Pe,
    linewidth=1.5
)

plt.axvline(
    disturbance_time,
    linestyle="--",
    label="Disturbance"
)

plt.xlabel("Time (s)")
plt.ylabel("Electrical Power Pe (pu)")
plt.title("Electrical Power Response")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "electrical_power_response.png",
    dpi=300
)

plt.show()
# ============================================================
# 13. DAMPING SENSITIVITY ANALYSIS
# ============================================================
D_values = [
    0.0,
    0.5,
    1.0,
    2.0,
    4.0
]
print("\n" + "=" * 60)
print("DAMPING SENSITIVITY ANALYSIS")
print("=" * 60)
sensitivity_results = []
plt.figure(figsize=(9, 5))
for D_test in D_values:

    def model_for_D(t, x):

        if t < disturbance_time:
            Pm = Pm_initial
        else:
            Pm = Pm_disturbed

        return generator_model(
            t,
            x,
            Pm=Pm,
            D_value=D_test
        )
    sol_D = solve_ivp(
        model_for_D,
        (t_start, t_end),
        x0,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-10
    )
    delta_D = sol_D.y[0]

    # Calculate Jacobian for this damping value
    def equilibrium_for_D(x):

        return generator_model(
            0.0,
            x,
            Pm=Pm_initial,
            D_value=D_test
        )


    A_D = numerical_jacobian(
        equilibrium_for_D,
        x0
    )


    eig_D = eigvals(A_D)
    # Dominant mode = eigenvalue with largest real part
    dominant_index = np.argmax(
        np.real(eig_D)
    )

    dominant_eigenvalue = eig_D[
        dominant_index
    ]

    sigma = np.real(
        dominant_eigenvalue
    )

    omega = np.imag(
        dominant_eigenvalue
    )

    magnitude = np.abs(
        dominant_eigenvalue
    )
    if magnitude > 1e-12:

        zeta = -sigma / magnitude

    else:

        zeta = np.nan
    sensitivity_results.append(
        (
            D_test,
            dominant_eigenvalue,
            zeta
        )
    )
    plt.plot(
        sol_D.t,
        np.rad2deg(delta_D),
        label=f"D = {D_test}"
    )
plt.axvline(
    disturbance_time,
    linestyle="--"
)

plt.xlabel("Time (s)")
plt.ylabel("Rotor Angle δ (degrees)")
plt.title("Damping Sensitivity: Rotor Angle")

plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "damping_sensitivity.png",
    dpi=300
)

plt.show()
# Print sensitivity table
print(
    f"{'D':>8}"
    f"{'Dominant Eigenvalue':>25}"
    f"{'Damping Ratio':>20}"
)

for D_test, eig_D, zeta in sensitivity_results:

    print(
        f"{D_test:8.2f}"
        f"{str(eig_D):>25}"
        f"{zeta:20.6f}"
    )
# ============================================================
# 14. DISCRETE STATE-SPACE REGRESSION
# ============================================================
#
# We use simulated data to estimate:
#
# x[k+1] = Ad x[k]
#
# This is a simple grey-box/system-identification step.
#
# ============================================================

# Use data after the disturbance
X = solution.y.T

# Sampling interval
dt = time[1] - time[0]


# Current state matrix
X_k = X[:-1, :]

# Next state matrix
X_k1 = X[1:, :]
# Least-squares estimate
Ad_estimated = np.linalg.lstsq(
    X_k,
    X_k1,
    rcond=None
)[0].T


print("\n" + "=" * 60)
print("ESTIMATED DISCRETE STATE MATRIX")
print("=" * 60)

print(Ad_estimated)
# ============================================================
# 15. DISCRETE EIGENVALUES
# ============================================================
discrete_eigenvalues = eigvals(
    Ad_estimated
)


print("\n" + "=" * 60)
print("DISCRETE-TIME EIGENVALUES")
print("=" * 60)

for i, z in enumerate(
    discrete_eigenvalues
):

    print(
        f"Mode {i + 1}: {z}"
    )
# Convert discrete eigenvalues to continuous-time estimates
continuous_estimated = (
    np.log(discrete_eigenvalues) / dt
)
print("\n" + "=" * 60)
print("ESTIMATED CONTINUOUS-TIME MODES")
print("=" * 60)

for i, lam in enumerate(
    continuous_estimated
):

    print(
        f"Mode {i + 1}: {lam}"
    )
# ============================================================
# 16. BASIC PRONY ANALYSIS
# ============================================================
#
# Prony analysis models a signal as:
#
# y[k] = sum(C_i * z_i^k)
#
# where z_i are discrete-time modal roots.
#
# For simplicity, we use a real-valued scalar signal:
# rotor-angle deviation.
#
# ============================================================

def prony_analysis(signal, dt, order=4):
    """
    Basic Prony-style analysis.

    Parameters
    ----------
    signal : 1D numpy array
        Input signal.

    dt : float
        Sampling interval.

    order : int
        Model order.

    Returns
    -------
    poles : complex array
        Discrete poles.

    continuous_poles : complex array
        Continuous-time poles.
    """
    y = np.asarray(
        signal,
        dtype=float
    )
    # Remove mean
    y = y - np.mean(y)

    N = len(y)

    if N <= 2 * order:

        raise ValueError(
            "Signal is too short for selected order."
        )
    # Build prediction matrix
    rows = N - order

    H = np.zeros(
        (rows, order)
    )

    target = np.zeros(
        rows
    )
    for i in range(rows):

        H[i, :] = y[
            i:i + order
        ][::-1]

        target[i] = y[
            i + order
        ]
    # Solve:
    #
    # y[n] + a1*y[n-1] + ... + ap*y[n-p] = 0
    #
    coefficients = np.linalg.lstsq(
        H,
        target,
        rcond=None
    )[0]
    # Characteristic polynomial:
    #
    # z^p - a1*z^(p-1) - ... - ap = 0
    #
    polynomial = np.concatenate(
        (
            [1.0],
            -coefficients
        )
    )
    poles = np.roots(
        polynomial
    )
    # Continuous-time poles
    continuous_poles = (
        np.log(poles) / dt
    )
    return poles, continuous_poles
# Use rotor-angle deviation
signal = (
    np.rad2deg(delta)
    - np.rad2deg(delta[0])
)


# To make Prony analysis focus on the response,
# use a section after the disturbance.
start_index = np.searchsorted(
    time,
    disturbance_time
)
signal_prony = signal[
    start_index:
]

# Use a relatively small order
prony_order = 4
prony_poles, prony_continuous = (
    prony_analysis(
        signal_prony,
        dt,
        order=prony_order
    )
)
print("\n" + "=" * 60)
print("PRONY ANALYSIS")
print("=" * 60)

for i, (z, lam) in enumerate(
    zip(
        prony_poles,
        prony_continuous
    )
):

    sigma = np.real(lam)
    omega = np.imag(lam)
    magnitude = np.abs(lam)
    if magnitude > 1e-12:

        zeta = -sigma / magnitude
    else:
        zeta = np.nan
    frequency = (
        abs(omega) / (2.0 * np.pi)
    )
    print(f"\nMode {i + 1}")
    print(f"Discrete pole      = {z}")
    print(f"Continuous pole    = {lam}")
    print(f"Damping ratio      = {zeta}")
    print(
        f"Frequency          = "
        f"{frequency} Hz"
    )
# ============================================================
# 17. SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT SUMMARY")
print("=" * 60)

print("""
The project implemented:

1. Third-order nonlinear synchronous-generator model
2. Generator connected to an infinite bus
3. Equilibrium operating-point calculation
4. Numerical Jacobian linearization
5. Eigenvalue-based small-signal stability analysis
6. Damping-ratio calculation
7. Oscillation-frequency calculation
8. Nonlinear disturbance simulation
9. Damping sensitivity analysis
10. Discrete state-space regression
11. Prony spectral analysis

The main state variables were:

    delta       -> rotor angle
    Delta omega -> speed deviation
    Eq'         -> transient EMF
""")
