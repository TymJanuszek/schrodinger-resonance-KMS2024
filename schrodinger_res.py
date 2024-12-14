import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def breit_wigner(x, M, G):
    return G / ((x - M) ** 2 + G ** 2 / 4)


class Model:

    def __init__(self, bins=100, dt=0.0001, duration=50, sampling=100, kappa=0, omega=0):
        """
        Init function
        :param bins: number of bins on X axis
        :param kappa: drive magnitude
        :param omega: drive angular frequency
        :param dt: time step
        :param duration: simulation strength
        :param sampling: sampling rate of energy, norm and position
        """
        self.dx = 1 / bins
        self.bins = bins

        self.X = np.arange(0, 1 + self.dx / 2, self.dx)
        self.rePsi = np.sqrt(2) * np.sin(np.pi * self.X)
        self.rePsi[0] = self.rePsi[-1] = 0
        self.imPsi = np.zeros(bins + 1)

        self.sampling = sampling

        self.dt = dt
        self.duration = duration
        self.time = np.arange(0, self.duration, self.dt)
        self.energy = np.zeros(int(len(self.time) / self.sampling))

        self.kappa = kappa
        self.omega = omega

    @staticmethod
    def subplot_grid(size):
        if size > 2:
            return 2, round(size / 2)
        else:
            return 1, 2

    def count_NxE(self, tau):
        N = self.dx * np.sum(self.rePsi ** 2 + self.imPsi ** 2)

        x = self.dx * np.sum(self.X * (self.rePsi ** 2 + self.imPsi ** 2))

        rePsiHam = self.rePsi * self.count_Re_hamilton(tau, self.rePsi)
        imPsiHam = self.imPsi * self.count_Im_hamilton(tau, self.imPsi)
        E = self.dx * np.sum(rePsiHam + imPsiHam)

        return N, x, E

    def energy_plot_save(self, filename):
        fig = plt.figure(figsize=(15, 9))
        plt.title(filename)
        time = np.arange(0, self.duration, self.sampling * self.dt)

        plt.plot(time, self.energy, "r")
        plt.xlabel("Time")
        plt.ylabel("Energy")
        plt.savefig(filename)

    def energy_plot_save_args(self, filename, titles, *args):
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle(filename)
        time = np.arange(0, self.duration, self.sampling * self.dt)

        i = 0
        hei, wid = self.subplot_grid(len(args))

        for arg in args:
            ax = plt.subplot(hei, wid, i + 1)
            ax.plot(time, arg)
            ax.set_title(titles[i])
            plt.xlabel("Time")
            plt.ylabel("Energy")
            i += 1

        plt.savefig(filename)

    def energy_plot_save_arr(self, filename, titles, energies):
        fig = plt.figure(figsize=(15, 9))
        fig.suptitle(filename)
        time = np.arange(0, self.duration, self.sampling * self.dt)

        i = 0
        hei, wid = self.subplot_grid(len(energies))

        for arg in energies:
            ax = plt.subplot(hei, wid, i + 1)
            ax.plot(time, arg)
            ax.set_title(titles[i])
            plt.xlabel("Time")
            plt.ylabel("Energy")
            i += 1

        plt.savefig(filename)

    def resonance_plot_n_fit(self, maxenergies, omegas, filename):
        fig = plt.figure(figsize=(15, 9))
        print(maxenergies)
        print(omegas)
        popt, pcov = curve_fit(breit_wigner, omegas, maxenergies)
        print(popt)
        print(pcov)

        omegas_cont = np.arange(np.min(omegas), np.max(omegas), 0.01)
        plt.plot(omegas_cont, breit_wigner(omegas_cont, *popt) + np.min(maxenergies), "r")
        plt.scatter(omegas, maxenergies)
        plt.xlabel(r"\omega")
        plt.ylabel(r"E_max")
        plt.savefig(filename)

    def count_Im_hamilton(self, tau, imPsi):
        HIm = np.zeros(self.bins + 1)

        for k in range(1, self.bins):
            HIm[k] = -0.5 * (imPsi[k + 1] + imPsi[k - 1] - 2 * imPsi[k]) / (self.dx ** 2) + self.kappa * (
                    self.X[k] - 0.5) * imPsi[k] * np.sin(self.omega * tau)

        HIm[0] = HIm[-1] = 0
        return HIm

    def count_Re_hamilton(self, tau, rePsi):
        HRe = np.zeros(self.bins + 1)

        for k in range(1, self.bins):
            HRe[k] = -0.5 * (rePsi[k + 1] + rePsi[k - 1] - 2 * rePsi[k]) / (self.dx ** 2) + self.kappa * (
                    self.X[k] - 0.5) * rePsi[k] * np.sin(self.omega * tau)

        HRe[0] = HRe[-1] = 0
        return HRe

    def simulate(self, silence):
        print("Starting simulation for", self.omega, self.kappa)

        for j in range(0, len(self.time)):
            tau = self.time[j]
            if j % self.sampling == 0:
                print(round(tau, 2))
                i = int(j / 100)
                N, x, self.energy[i] = self.count_NxE(tau)
                if not silence:
                    print("    N = ", N)
                    print("    x = ", x)
                    print("    E = ", self.energy[i])

            rePsi_halftau = self.rePsi + self.count_Im_hamilton(tau, self.imPsi) * 0.5 * self.dt
            self.imPsi = self.imPsi - self.count_Re_hamilton(tau + 0.5 * self.dt, rePsi_halftau) * self.dt
            self.rePsi = rePsi_halftau + self.count_Im_hamilton(tau + self.dt, self.imPsi) * 0.5 * self.dt

        return self.energy

    def animate(self):
        for j in range(0, len(self.time)):
            tau = self.time[j]

            if j % self.sampling == 0:
                i = int(j / 100)
                print(round(tau, 2))
                N, x, self.energy[i] = self.count_NxE(tau)
                print("    N = ", N)
                print("    x = ", x)
                print("    E = ", self.energy[i])

            rePsi_halftau = self.rePsi + self.count_Im_hamilton(tau, self.imPsi) * 0.5 * self.dt
            self.imPsi = self.imPsi - self.count_Re_hamilton(tau + 0.5 * self.dt, rePsi_halftau) * self.dt
            self.rePsi = rePsi_halftau + self.count_Im_hamilton(tau + self.dt, self.imPsi) * 0.5 * self.dt

            if j % 10 == 0:
                ax = plt.gca()
                ax.set_ylim([-2, 2])
                plt.plot(self.X, self.imPsi, "b", label="Im(Psi)")
                plt.plot(self.X, self.rePsi, "r", label="Re(Psi)")
                plt.legend()
                plt.draw()
                plt.pause(0.0001)
                plt.clf()
