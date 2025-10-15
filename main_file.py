#Name- Eklavya Chauhan
#Roll No- 2311067

import math
from mylib import random_number, NewtonRaphson, LUdecomposition, GaussSeidel

#=============Q1: Monte Carlo estimation of ellipse area=============

OUTPUT_FILE1 = "output_Q1.txt"
open(OUTPUT_FILE1, "w").close()  # clear previous content

def log(text="\n"):
    """Print to screen and write instantly to file."""
    print(text)
    with open(OUTPUT_FILE1, "a") as f:
        f.write(str(text) + "\n")
        f.flush()   

a = 2  # semi-major axis
b = 1  # semi-minor axis
N = 50000  # number of random points (enough for <5% error)

rng = random_number(seed=7)
inside = 0

for _ in range(N):
    x = 4 * rng.next_float() - 2   # range [-2, 2]
    y = 2 * rng.next_float() - 1   # range [-1, 1]
    if (x**2 / a**2 + y**2 / b**2) <= 1:
        inside += 1

# Compute results
area_box = 4 * 2  # width × height of bounding box
area_est = (inside / N) * area_box
area_exact = math.pi * a * b
error = abs(area_est - area_exact) / area_exact * 100

# Write results
log("Q1. Monte Carlo Estimation of Ellipse Area")
log(f"Semi-major axis (a) = {a}")
log(f"Semi-minor axis (b) = {b}")
log(f"Random points used  = {N}")
log(f"Estimated area      = {area_est:.6f}")
log(f"Analytical area     = {area_exact:.6f}")
log(f"Percentage error    = {error:.3f}%")
log("=" * 80)

#=============Q2: Newton-Raphson method for Wien's displacement law=============

# Clear output file
OUTPUT_FILE2 = "output_Q2.txt"
open(OUTPUT_FILE2, "w").close()

def log(text="\n"):
    print(text)
    with open(OUTPUT_FILE2, "a") as f:
        f.write(str(text) + "\n")

# Constants
h = 6.626e-34       # Planck's constant (J·s)
k = 1.381e-23       # Boltzmann constant (J/K)
c = 3e8             # Speed of light (m/s)

# Define function and its derivative
def f(x):
    return (x - 5) * math.exp(x) + 5

def df(x):
    return math.exp(x) * (x - 4)

# Solve using Newton–Raphson
solver = NewtonRaphson(f, df, x0=5.0, tol=1e-6, outfile=OUTPUT_FILE2)
root, iters = solver.solve()

# Compute Wien’s constant
b = h * c / (k * root)

# Display and save results
log("Q2. Wien’s displacement law using Newton–Raphson method")
log("Equation solved: (x - 5)e^x + 5 = 0")
log(f"Initial guess = 5.0")
log(f"Converged root (x) = {root:.6f}")
log(f"Iterations = {iters}")
log(f"Wien’s constant b = {b:.6e} m·K")
log("=" * 80)

#=============Q3: LU Decomposition=============

OUTPUT_FILE3 = "output_Q3.txt"
open(OUTPUT_FILE3, "w").close()

try:
    LUdecomposition("matrix_Q3.txt", outfile=OUTPUT_FILE3)
except Exception as e:
    with open("output.txt", "a") as f:
        f.write(f"Error: {e}\n")
    print(f"Error: {e}")

#=============Q4: Gauss-Seidel Iterative Method=============

OUTPUT_FILE = "output_Q4.txt"
open(OUTPUT_FILE, "w").close()  # clear previous output

def log(text="\n"):
    print(text)
    with open(OUTPUT_FILE, "a") as f:
        f.write(str(text) + "\n")

log("Q4. Gauss-Seidel Iterative Method\n")
log("Matrix A read from: matrixA.txt")
log("Vector b read from: vectorB.txt\n")

try:
    # ✅ Pass filenames positionally
    solver = GaussSeidel("matrixA.txt", "vectorB.txt", tol=1e-6, max_steps=1000)
    solution = solver.solve()

    log("\nFinal Solution Vector:")
    for i, val in enumerate(solution):
        log(f"x{i+1} = {val:.6f}")
    log("=" * 80)

except Exception as e:
    log(f"Error: {e}")
    log("=" * 80)
