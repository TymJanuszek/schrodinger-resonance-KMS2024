import schrodinger_res
import numpy as np

omega = 3 * np.pi ** 2 / 2
duration = 50

model = schrodinger_res.Model(duration=duration, kappa=1, omega=omega)

percentage = np.arange(0.90, 1.12, 0.02)
per_len = len(percentage)
energies = np.zeros(shape=(per_len, len(model.energy)))

names = ["90%", "92%", "94%", "96%", "98%", "100%", "102%", "104%", "106%", "108%", "110%", "112%"]

for i in range(0, per_len):
    model = schrodinger_res.Model(duration=duration, kappa=1, omega=percentage[i]*omega)
    energies[i] = model.simulate(True)


model.plot_save_arr("90-96.png", names[0:4], energies[0:4])
model.plot_save_arr("98-104.png", names[4:8], energies[4:8])
model.plot_save_arr("106-112.png", names[8:12], energies[8:12])
