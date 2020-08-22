import os
import sys
import subprocess
import time
import pandas as pd
from main import run_file


def proc_file(path):
    p = subprocess.Popen('python3 controller/main.py ' + path, shell=True)
    p.wait()


if __name__ == "__main__":
    # Create_report folder
    if not os.path.exists('results/'):
        os.makedirs('results/')

    malware = pd.read_csv('data.csv')

    for line in malware['md5'].values[:5000]:
        continue_fl = False
        for _, _, files in os.walk('results/'):
            for file_ in files:
                if line in file_:
                    print('Found results!')
                    continue_fl = True

        if not continue_fl:
            path = 'psi_graph/malware/'+ line + '.txt'
            if os.path.exists(path):
                run_file(path)
            else:
                print('PSI not exists')
