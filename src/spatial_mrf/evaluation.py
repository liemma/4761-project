import matplotlib.pyplot as plt

def plot_energy_history(result):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(result.energy_history) + 1), result.energy_history, marker='o', color='#2c3e50')
    plt.title("AW-HMRF Convergence: Total Gibbs Energy")
    plt.xlabel("EM Iterations")
    plt.ylabel("Total Energy (Unary + Weighted Pairwise)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()