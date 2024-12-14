import schrodinger_res
import numpy as np
import matplotlib.pyplot as plt

omega = 3 * np.pi ** 2 / 2
duration = 50
model = schrodinger_res.Model(duration=duration, kappa=1, omega=omega)

omegas = omega * np.arange(0.90, 1.12, 0.02)
per_len = len(omegas)
energies = np.zeros(shape=(per_len, len(model.energy)))
maxenergies = np.zeros(12)

names = ["90%", "92%", "94%", "96%", "98%", "100%", "102%", "104%", "106%", "108%", "110%", "112%"]

for i in range(0, per_len):
    model = schrodinger_res.Model(duration=duration, kappa=1, omega=omegas[i])
    energies[i] = model.simulate(True)
    maxenergies[i] = np.max(energies[i])

model.energy_plot_save_arr("90-96.png", names[0:4], energies[0:4])
model.energy_plot_save_arr("98-104.png", names[4:8], energies[4:8])
model.energy_plot_save_arr("106-112.png", names[8:12], energies[8:12])

model.resonance_plot_n_fit(maxenergies, omegas, "resonance.png")
