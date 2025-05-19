import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

x = np.linspace(-4, 4, 1000)

# PDF for each distribution
normal_pdf = norm.pdf(x, loc=0, scale=1)  # N(0,1)
t1_pdf = t.pdf(x, df=1)
t5_pdf = t.pdf(x, df=5)
t10_pdf = t.pdf(x, df=10)

plt.figure(figsize=(10, 6))

plt.plot(x, normal_pdf, label='N(0, 1)', color='blue', linestyle='--')
plt.plot(x, t1_pdf, label='t(1)', color='red')
plt.plot(x, t5_pdf, label='t(5)', color='green')
plt.plot(x, t10_pdf, label='t(10)', color='orange')

plt.title('Densité des lois normale et and de student')
plt.xlabel('x')
plt.ylabel('Densité')
plt.legend()

plt.grid(True)
plt.show()