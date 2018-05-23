import os
import numpy as np

def distance(folder, name, distance_level):
    file_distance = folder+name+"_distance.txt"
    if os.path.isfile(file_distance):
        with open(file_distance, 'r') as f:
            distance = float(f.read().splitlines()[0])
        return distance <= distance_level
    else:
        return False

def select_networks(directory, distance_level=0.2):
    networks = np.empty((3, 0), dtype=np.object)
    for name in os.listdir(directory):
        folder = directory+os.fsdecode(name)+'/'
        if os.path.isdir(folder):
            if distance(folder, name, distance_level):
                network = folder+name+"_porepy.csv"
                network_topo = folder+name+"_topo_porepy.csv"
                if os.path.isfile(network) and os.path.isfile(network_topo):
                    files = np.array([name, network, network_topo]).reshape((3, 1))
                    networks = np.append(networks, files, axis=1)
    return networks

if __name__ == "__main__":
    select_networks('./networks/')
