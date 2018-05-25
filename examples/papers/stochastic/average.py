import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from select_networks import select_networks

def main():
    N = 1440 # number of time steps

    networks_id = select_networks('./networks/')[0, :]
    prod = np.empty((N, networks_id.size)) # production data
    prod_topo = np.empty(prod.shape)

    # load all the data
    for idx, name in enumerate(networks_id):
        network = 'result_'+name+'/production.txt'
        data = np.loadtxt(network, delimiter=',', unpack=True)
        prod[:, idx] = data[:, 1]

        network_topo = 'result_'+name+'_topo/production.txt'
        data = np.loadtxt(network_topo, delimiter=',', unpack=True)
        prod_topo[:, idx] = data[:, 1]

    stat = np.empty((N, 3)) #average and lower and upper percentiles
    stat_topo = np.empty(stat.shape) #average and lower and upper percentiles

    # compute the statistics information
    perc = [10, 90]
    stat[:, 0] = np.mean(prod, axis=1)
    stat[:, 1:] = np.percentile(prod, perc, axis=1).T

    stat_topo[:, 0] = np.mean(prod_topo, axis=1)
    stat_topo[:, 1:] = np.percentile(prod_topo, perc, axis=1).T

    # load the production from the original network
    original = 'result_original/production.txt'
    prod_orig = np.loadtxt(original, delimiter=',', unpack=True)[:, 1]

    # the time is the same, read it just once
    year = np.pi*1e7
    time = np.loadtxt(network, delimiter=',', unpack=True)[:, 0]/year

    # setup the plot
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=15)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('year')
    ax.set_ylabel('production')

    perc = list(map(str, perc))
    ax.plot(time, stat[:, 0], color='r', label='av')
    ax.plot(time, stat[:, 1], color='r', label=perc[0]+'\%', dashes=[6, 2])
    ax.plot(time, stat[:, 2], color='r', label=perc[1]+'\%', dashes=[6, 2])
    ax.fill_between(time, stat[:, 1], stat[:, 2], facecolor='r', alpha=0.2)

    ax.plot(time, stat_topo[:, 0], color='b', label='av-t')
    ax.plot(time, stat_topo[:, 1], color='b', label=perc[0]+'\%-t', dashes=[6, 2])
    ax.plot(time, stat_topo[:, 2], color='b', label=perc[1]+'\%-t', dashes=[6, 2])
    ax.fill_between(time, stat_topo[:, 1], stat_topo[:, 2], facecolor='b', alpha=0.2)

    ax.plot(time, prod_orig, color='g', label='or', lw=2)

    ax.legend()
    plt.show()

    figure_name = "prod_comparison.pdf"
    with PdfPages(figure_name) as pdf:
        pdf.savefig(fig)

if __name__ == "__main__":
    main()
