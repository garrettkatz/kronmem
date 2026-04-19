import numpy as np
import matplotlib.pyplot as pt

x = np.linspace(-3, 3, 100)
pt.plot(x, np.where(x > 0, x**(1/3), -(-x)**(1/3)), 'k:', label=r"$x^{(1/3)}$")
pt.plot(x, np.where(x > 0, x/3 + 2/3, x/3 - 2/3), 'k-', label=r"$\sigma_3(x)$")
pt.xlabel(r"$x$")
pt.legend()
pt.savefig("sigma.pdf")
pt.show()

