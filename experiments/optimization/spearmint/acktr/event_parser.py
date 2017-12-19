import os
from os import listdir
from os.path import isfile, join
import tensorflow as tf
csv_dir = '/tmp/rosrl/GazeboModularScara3DOFv3Env/acktr/event_data.csv' 
csv = open(csv_dir, "w") 
num_event_files = 22
for count in range(1,num_event_files):
        outdir='/tmp/rosrl/GazeboModularScara3DOFv3Env/acktr/'+str(count)+'/error/'
        for f in listdir(outdir):
              file_path = outdir + str(f)
              print("file_path", file_path)
        for e in tf.train.summary_iterator(file_path):
                for v in e.summary.value:
                        if v.tag == 'Simulation rewards':
                                 print(v.simple_value)
                                 row = str(v.simple_value) + "\n"
                                 csv.write(row)
