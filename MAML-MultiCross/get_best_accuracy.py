import numpy as np

# with open('201646.out', 'r') as f: # 0.6547619047619048
# with open('201648.out', 'r') as f: # 0.6547619047619048
# with open('201647.out', 'r') as f: # 0.7142857142857143
# with open('201758.out', 'r') as f: # 0.6309523809523809 This is the run after fixing bug in labeling, choosing the 'best' model
# with open('201759.out', 'r') as f: # Best accuracy: 0.6547619047619048 This is the run after fixing bug in labeling, choosing the 'latest' model
with open('201762.out', 'r') as f: # Best accuracy: 0.6666666666666666 This is the run after fixing bug in labeling, choosing the 'latest' model (1st 3 experiments)
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