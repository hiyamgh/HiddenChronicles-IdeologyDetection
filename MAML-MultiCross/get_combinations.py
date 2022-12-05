# get all combinations 1-12 of 2
# get all combinations 1-12 (2) including English in all
# get all combinations 1-12 (2) including Arabic in all

from itertools import combinations

langs = [str(n) for n in range(1, 13) if n != 3]
comb2 = list(combinations(langs, 2))
comb2_en = []
for comb in comb2:
    if "12" in comb:
        pass
    else:
        c = list(comb)
        c.append("12")
        c = tuple(c)
        comb2_en.append(c)

print(len(comb2))
print(len(comb2_en))

with open('experiments_langs.txt', 'w') as f:
    exp_str = '--meta_update_method threewayprotomaml --train_datasets_ids {} --dev_dataset_id 3 --test_dataset_id 14 --total_epochs 5 --total_iter_per_epoch 100 --total_epochs_before_pause 10 --init_inner_loop_learning_rate 4e-5 --meta_learning_rate 3e-5 --meta_inner_optimizer_learning_rate 5e-5'
    for comb in comb2:
        f.write(exp_str.format('{},{}'.format(comb[0], comb[1])) + "\n")

# with open('experiments_langs3.txt', 'w') as f:
#     exp_str = '--meta_update_method threewayprotomaml --train_datasets_ids {} --dev_dataset_id 3 --test_dataset_id 14 --total_epochs 5 --total_iter_per_epoch 100 --total_epochs_before_pause 10 --init_inner_loop_learning_rate 4e-5 --meta_learning_rate 3e-5 --meta_inner_optimizer_learning_rate 5e-5'
#     for comb in comb2_en:
#         f.write(exp_str.format('{},{},{}'.format(comb[0], comb[1], comb[2])) + "\n")