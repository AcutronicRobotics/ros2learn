import os
import csv
import time
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from collections import defaultdict
from scipy.signal import savgol_filter

def plot_results(plot_title, all_values, labels, num_timesteps, y_lim, smooth, color_defaults):
    lines = []
    names = []

    for i in range( len(all_values) ):
        columns = defaultdict(list)

        with open( all_values[i] ) as f:
            reader = csv.DictReader(f) # read rows into a dictionary format
            for row in reader: # read a row as {column1: value1, column2: value2,...}
                for (k,v) in row.items(): # go over each column name and value
                    if v is '':
                        v = 'nan'
                    columns[k].append(v) # append the value into the appropriate list based on column name k

        y_mean = np.asarray( list( map(float,columns['eprewmean']) ) )
        y_std = np.asarray( list( map(float,columns['eprewsem']) ) )

        x = np.linspace(0, num_timesteps, y_mean.size, endpoint=True)

        if smooth is True:
            y_mean = savgol_filter(y_mean, 11, 3)
            y_std = savgol_filter(y_std, 11, 3)

        # print("i: ", i, "; y_mean_max: ", max(y_mean), "; y_mean_min: ", min(y_mean), "; overall mean: ", np.mean(y_mean), "; overall_std: ", np.std(y_mean))

        y_upper = y_mean + y_std
        y_lower = y_mean - y_std

        color = color_defaults[i]

        # if labels[i] == 'ACKTR':
        #     plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=1)
        # else:
        #     plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.4)

        plt.fill_between(x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.4)
        line = plt.plot(x, list(y_mean), color=color, rasterized=False, antialiased=True)

        lines.append(line[0])
        names.append(labels[i])

    plt.legend(lines, names, loc=4) # lower right
    plt.xlim([0, num_timesteps])
    plt.ylim(y_lim)
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Mean Episode Reward")
    plt.title(plot_title)
    plt.xticks([200000, 400000, 600000, 800000, 1000000], ["200K", "400K", "600K", "800K", "1M"])

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-env', '--env_id', help='title of the plot', type=str, default="MARA-v0")
    parser.add_argument('-dirs', '--directories', help='list of directories to the progress csv files', nargs='+', type=str, required=True)
    parser.add_argument('-l', '--labels', help='list of labels (algorithms) of the plot', nargs='+', type=str, choices=['PPO', 'TRPO', 'ACKTR'], required=True)
    parser.add_argument('-ts', '--num_timesteps', help='maximum x-axis limit', type=int, default=1000000)
    parser.add_argument('-min_mer', '--ymin', help='minimum y-axis limit (Mean Episode Reward)', type=int, default=-2100)
    parser.add_argument('-max_mer', '--ymax', help='maximum y-axis limit (Mean Episode Reward)', type=int, default=0)
    args = parser.parse_args()

    assert len(args.labels) <= 10
    assert len(args.directories) <= 10
    assert len(args.directories) == len(args.labels)

    color_defaults = [
        '#2ca02c',  # cooked asparagus green
        '#1f77b4',  # muted blue
        '#ff7f0e',  # safety orange
        '#d62728',  # brick red
        '#9467bd',  # muted purple
        '#8c564b',  # chestnut brown
        '#e377c2',  # raspberry yogurt pink
        '#7f7f7f',  # middle gray
        '#bcbd22',  # curry yellow-green
        '#17becf']  # blue-teal

    matplotlib.rcParams.update({'font.size': 12})

    plot_results(args.env_id, args.directories, args.labels, args.num_timesteps, [args.ymin, args.ymax], True, color_defaults)
    plt.tight_layout()

    savedir = '/tmp/ros2learn/plots/'
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plt.savefig( savedir + args.env_id + '.png', dpi=400, facecolor='w', edgecolor='w',
            orientation='landscape', papertype='b0', format=None,
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)

    plt.show()

if __name__ == '__main__':
    main()
