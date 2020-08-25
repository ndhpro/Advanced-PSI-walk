import os
import pandas as pd
import shutil


root = "psi-walk/malware/"
df = pd.read_csv("data.csv")
known = [
    "Gafgyt/BASHLITE (2014)",
    "Mirai (2016)",
    "Tsunami/Kaiten (2013)",
    "Spike/Dofloo/MrBlack (2014)",
    "LightAidra/Aidra (2012)",
    "Darlloz (2013)"
]
for _, _, files in os.walk(root):
    for fname in files:
        label = df.loc[df["md5"]==fname[:fname.find(".txt")], "label"].values[0]
        if label not in known:
            print(fname, label)
            f_path = root + fname
            shutil.move(f_path, "psi-walk/test_unk")