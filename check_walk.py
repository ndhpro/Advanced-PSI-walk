import os


file_list = []
for root in ["psi-walk/malware/", "psi-walk/benign/"]:
    for _, _, files in os.walk(root):
        for file_name in files:
            file_list.append(root + file_name)

c = 0
for fx in file_list:
    for fy in file_list:
        if os.path.getsize(fx) == os.path.getsize(fy):
            with open(fx, 'r') as f:
                dx = f.read()
            with open(fy, 'r') as f:
                dy = f.read()
            if dx == dy:
                print(fx, fy)
                c += 1
print(c)
