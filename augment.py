import os
import random
random.seed(2020)


psi_path = ["psi_walk_v2.5/malware/", "psi_walk_v2.5/benign/"]

for root in psi_path:
    for _, _, files in os.walk(root):
        prev_lines = []
        for fname in files:
            with open(root + fname, 'r') as f:
                lines = f.readlines()

            # shuffle
            random.shuffle(lines)
            with open(root.replace("psi_walk_v2.5", "aug") + fname + "0", 'w') as f:
                for line in lines:
                    f.write(line)

            # randomly delete
            if len(lines) > 1:
                idx = random.randrange(len(lines))
                lines_ = lines.copy()
                del lines_[idx]
                with open(root.replace("psi_walk_v2.5", "aug") + fname + "1", 'w') as f:
                    for line in lines_:
                        f.write(line)

            # mix
            mix = prev_lines + random.choices(lines, k=int((len(lines)+1)/2))
            random.shuffle(mix)
            with open(root.replace("psi_walk_v2.5", "aug") + fname + "2", 'w') as f:
                for line in mix:
                    f.write(line)
            prev_lines = random.choices(lines, k=int((len(lines)+1)/2))
