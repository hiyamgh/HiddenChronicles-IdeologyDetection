import numpy as np
with open('201647.out', 'r') as f:
    best_acc = 0.0
    lines = f.readlines()
    for line in lines:
        if 'ccuracy' in line:
            acc = np.float(line.split(' ')[1].strip())
            if acc == 0.02877:
                print()
            if acc > best_acc:
                best_acc = acc

print('Best accuracy: {}'.format(best_acc))