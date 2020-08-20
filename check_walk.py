import os


PATH = "psi_walk/malware/"
c = 0
for _, _, files in os.walk(PATH):
    for file in files:
        try:
            with open(PATH + file, 'r') as f:
                dnew = f.read()
            with open("D:/Projects/PSI-walk/datav2/malware/" + file, 'r') as f:
                dold = f.read()
        except Exception as e:
            print(e)
        if dnew != dold:
            c += 1
            print(file)
print(c)
