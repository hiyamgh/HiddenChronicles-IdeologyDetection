import numpy as np

# with open('201646.out', 'r') as f: # 0.6547619047619048
# with open('201648.out', 'r') as f: # 0.6547619047619048
with open('201647.out', 'r') as f: # 0.7142857142857143
    accuracies = []
    best_acc = 0.0
    lines = f.readlines()
    for line in lines:
        if 'ccuracy' in line:
            acc = np.float(line.split(' ')[1].strip())
            if acc == 0.02877:
                print()
            if acc > best_acc:
                best_acc = acc
            accuracies.append(acc)

print('Best accuracy: {}'.format(best_acc))
for acc in sorted(accuracies):
    print(acc)