import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


folder_name = ['result_25366', 'result_25366_topo']
figure_name = 'production.pdf'

color = ['b', 'r']
legend = ['no topo', 'topo']
year = np.pi*1e7
scale = 1

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=15)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('year')
ax.set_ylabel('production')

for f, c, l in zip(folder_name, color, legend):
    data = np.loadtxt(f+"/production.txt", delimiter=',', unpack=True)
    print(data.shape)
    ax.plot(data[:, 0]/year, data[:, 1]/scale, color=c, label=l)

ax.legend()
plt.show()
#plt.savefig(figure_name)

with PdfPages(figure_name) as pdf:
    pdf.savefig(fig)
