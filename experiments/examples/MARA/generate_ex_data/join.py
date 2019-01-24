import numpy as np
import os

save_path = "/home/yue/experiments/expert_data/mara/collisions_model"

def join(a):
    obsarray = []
    acsarray = []
    for e in range (0, len(a)):
        for traj in range (0, len(a[e].f.obs)):
            obsarray.append(a[e].f.obs[traj])
            acsarray.append(a[e].f.acs[traj])

    obsdata = np.array(obsarray)
    acsdata = np.array(acsarray)

    np.savez( os.path.join( save_path + '/expert_data.npz'), obs=obsdata, acs=acsdata)

ed1 = np.load(save_path + '/1_10_new.npz')
ed2 = np.load(save_path + '/expert_data1.npz')
ed3 = np.load(save_path + '/expert_data2.npz')
ed4 = np.load(save_path + '/expert_data3.npz')
ed5 = np.load(save_path + '/expert_data4.npz')
ed6 = np.load(save_path + '/expert_data5.npz')
ed7 = np.load(save_path + '/expert_data6.npz')
ed8 = np.load(save_path + '/expert_data7.npz')
ed9 = np.load(save_path + '/expert_data8.npz')
ed10 = np.load(save_path + '/expert_data9.npz')

data = [ed1, ed2, ed3, ed4, ed5, ed6, ed7, ed8, ed9, ed10]

join(data)

ed = np.load(save_path + '/expert_data.npz')
print(ed.f.obs.shape)
print(ed.f.acs.shape)
