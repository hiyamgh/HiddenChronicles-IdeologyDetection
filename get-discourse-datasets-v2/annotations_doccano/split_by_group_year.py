import pickle
import pandas as pd
import os

# get the sentences for PISL
# get the sentences for SSM
# pick one for dev and one for testing (in order to check the model's ability to capture the shifts
# across time)


def mkdir(folder):
    ''' Create a directory, if it does not already exist '''
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    sentences = []
    years = []
    groups = []
    group2name = {
        'group 0': 'Palestinian Resistance South Lebanon',
        'group 1': 'Sabra and Shatila Massacre'
    }

    name2savename = {
        'Palestinian Resistance South Lebanon': 'palestinian_resistance_south_lebanon',
        'Sabra and Shatila Massacre': 'sabra_shatila_massacre'
    }

    group2sentences = {}

    with open('sentences.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()

        for line in lines:
            if line.strip() == '':
                continue
            if 'group' in line.strip():
                year, group = line.strip().split('-')
                year = int(year.strip())
                group = group.strip()
                group = group2name[group]

                if group not in group2sentences:
                    group2sentences[group] = {}

                if year not in group2sentences[group]:
                    group2sentences[group][year] = []

                continue

            if line.strip().isdigit():
                continue

            else:
                group2sentences[group][year].append(line.strip())

    for group in group2sentences:
        count = 0
        for year in group2sentences[group]:
            count += len(group2sentences[group][year])
        print('Number of sentences in {}: {}'.format(group, count))

    with open('groups2years_sentences.pickle', 'wb') as handle:
        pickle.dump(group2sentences, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # split the annotations by group

    dirs = ['translationsv2/van_dijk_speeches/', 'translationsv2/van_dijk_contexts/',
            'translationsv2/propaganda/', 'translationsv2/argumentation/']

    ar_datasets = ['our_corpus_van_dijk_speeches.xlsx', 'our_corpus_van_dijk_contexts.xlsx',
                   'our_corpus_propaganda.xlsx', 'our_corpus_argumentation.xlsx']
    for i, dir in enumerate(dirs):
        for file in os.listdir(dir):
            if '.xlsx' in file:
                df = pd.read_excel(os.path.join(dir, file))
                for group in group2sentences:

                    save_dir = os.path.join(dir, group)
                    mkdir(save_dir)

                    df_group = pd.DataFrame(columns=['Original', 'Sentence', 'label'])
                    all_group_sentences = [s for y in group2sentences[group] for s in group2sentences[group][y]]
                    for j, row in df.iterrows():
                        sentence = str(row['Original']).strip()
                        if sentence in all_group_sentences:
                            df_group = df_group.append({
                                'Original': row['Original'],
                                'Sentence': row['Sentence'],
                                'label': row['label']
                            }, ignore_index=True)

                    assert len(df_group) == len(all_group_sentences)
                    df_group.to_excel(os.path.join(save_dir, file), index=False)

        df_ar = pd.read_excel(ar_datasets[i])
        for group in group2sentences:

            save_dir = os.path.join(dir, group)
            mkdir(save_dir)

            df_group = pd.DataFrame(columns=['Sentence', 'label'])
            all_group_sentences = [s for y in group2sentences[group] for s in group2sentences[group][y]]
            for i, row in df_ar.iterrows():
                sentence = str(row['text']).strip()
                if sentence in all_group_sentences:
                    df_group = df_group.append({
                        'Sentence': row['text'],
                        'label': row['label']
                    }, ignore_index=True)
            df_group.to_excel(os.path.join(save_dir, 'our_corpus_ar.xlsx'), index=False)