import pandas as pd
import os

def write_obs(obs=None, path="csv/obs_file.csv", env_name=None):
    df = pd.DataFrame(obs).T
    with open(path, 'a+') as f:
        if os.stat(path).st_size == 0:
            obs_headers = ["ob" + str(i) for i in range(len(obs))]
            df.to_csv(f, header=obs_headers, index=False)
        else:
            df.to_csv(f, header=False, index=False)

def write_acs(acs=None, path="acs_file.csv"):
    df = pd.DataFrame(acs).T
    with open(path, 'a+') as f:
        if os.stat(path).st_size == 0:
            acs_headers = ['ac1', 'ac2', 'ac3', 'ac4', 'ac5','ac6']
            df.to_csv(f, header=acs_headers, index=False)
        else:
            df.to_csv(f, header=False, index=False)

def write_rew(rew=None, path="rew_file.csv"):
    df = pd.DataFrame(rew).T
    with open(path, 'a+') as f:
        if os.stat(path).st_size == 0:
            rew_header = ['rew']
            df.to_csv(f, header=rew_header, index=False)
        else:
            df.to_csv(f, header=False, index=False)
