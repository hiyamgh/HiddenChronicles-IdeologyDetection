import os
import pandas as pd


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)


# Our corpus translations to Argumentation
df_orig = pd.read_excel('our_corpus_argumentation.xlsx')
translation_dir = 'translationsv2/'
save_dir = 'translationsv2/argumentation/'
mkdir(save_dir)
for file in os.listdir(translation_dir):
    if '.xlsx' in file:
        df = pd.read_excel(os.path.join(translation_dir, file))
        df['label'] = list(df_orig['label'])
        df.to_excel(os.path.join(save_dir, file), index=False)


# Our corpus translations to Propaganda
df_orig = pd.read_excel('our_corpus_propaganda.xlsx')
translation_dir = 'translationsv2/'
save_dir = 'translationsv2/propaganda/'
mkdir(save_dir)
for file in os.listdir(translation_dir):
    if '.xlsx' in file:
        df = pd.read_excel(os.path.join(translation_dir, file))
        df['label'] = list(df_orig['label'])
        df.to_excel(os.path.join(save_dir, file), index=False)

# Our corpus translations to Van Dijk Contexts
df_orig = pd.read_excel('our_corpus_van_dijk_contexts.xlsx')
translation_dir = 'translationsv2/'
save_dir = 'translationsv2/van_dijk_contexts/'
mkdir(save_dir)
for file in os.listdir(translation_dir):
    if '.xlsx' in file:
        df = pd.read_excel(os.path.join(translation_dir, file))
        df['label'] = list(df_orig['label'])
        df.to_excel(os.path.join(save_dir, file), index=False)

# Our corpus translations to Van Dijk Soeeches
df_orig = pd.read_excel('our_corpus_van_dijk_speeches.xlsx')
translation_dir = 'translationsv2/'
save_dir = 'translationsv2/van_dijk_speeches/'
mkdir(save_dir)
for file in os.listdir(translation_dir):
    if '.xlsx' in file:
        df = pd.read_excel(os.path.join(translation_dir, file))
        df['label'] = list(df_orig['label'])
        df.to_excel(os.path.join(save_dir, file), index=False)