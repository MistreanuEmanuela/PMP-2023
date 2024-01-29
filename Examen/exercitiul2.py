import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import sympy as sp


def main():

    # incercam sa descoperim theta pt 4 valori, a 1 a 2, 3 si 4
    for a in range(1, 5):
        numar_aruncari_pana_la_stema = 5**a

        h = 1
        t = numar_aruncari_pana_la_stema - h

        # x reprezintă posibilele valori ale probabilității succesului (stema)
        x = np.linspace(0, 1, 10)

        # dist_geom
        true_posterior = stats.geom.pmf(numar_aruncari_pana_la_stema - 1, x)
        plt.plot(x, true_posterior, label="True posterior (geometric)")

        # aprox_patratica
        mean_q = {"p": (h + 1) / (h + t + 2)}
        std_q = 1 / np.sqrt(h + t + 2)
        plt.plot(x, stats.norm.pdf(x, mean_q["p"], std_q), label="Quadratic approximation")

        # estimare val
        theta = sp.symbols("theta")
        posterior = theta * (1 - theta) ** (h + t)
        derivative = sp.diff(posterior, theta)
        maxima_theta = sp.solve(derivative, theta)

        print(
            f"Valoarea lui theta care maximizează probabilitatea a posteriori: {maxima_theta}"
        )

        plt.legend(loc=0, fontsize=13)
        plt.title(f"Prima apariție a unei steme: heads = {h}, tails = {t}")
        plt.xlabel("θ", fontsize=14)
        plt.yticks([])
        plt.show()
# observam cu cat a creste cu atat theta scade

if __name__ =="__main__":
    main()