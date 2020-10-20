from glob import glob
import pandas as pd


df = pd.read_csv("elf_v2.csv")
fpaths = glob("psi_walk_v2.5/malware/*")
for fpath in fpaths:
    fname = fpath.split('/')[-1].replace(".txt", "")
    with open(fpath, 'r') as f:
        data = f.read()
    enc = True
    for word in data.split():
        if not "sub" in word:
            enc = False
            break
    if enc:
        continue

    print(df.loc[df["md5"]==fname, "label"].values[0])