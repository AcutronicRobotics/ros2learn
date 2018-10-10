import matplotlib.pyplot as plt
import pandas
import argparse
import os

#example: python3 plot_csv.py --files csv/ppo_det_acs.csv csv/ppo_det_obs.csv --headers ac1 ob1

# parser
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--files', help='csv files path', nargs='+', default=None)
parser.add_argument('--headers', help='headers name to show for each csv file', nargs='+', default=None)
args = parser.parse_args()

data = []
for csv_file in args.files:
    f = pandas.read_csv(csv_file)
    data.append(f)

headers = args.headers
for i in range(0, len(data)):
    plt.plot(data[i][headers[i]])

plt.grid()

if not os.path.exists("img"):
    os.makedirs(directory)
plt.savefig('img/a.png')

plt.show()
