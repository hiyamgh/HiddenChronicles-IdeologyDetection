import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# df = pd.read_csv('sentences_ocr_corrected_discourse_profiling.csv')
# # print(df['Label'].value_counts())
# df = df.dropna()
# # print(any(np.nan(df['Sentence'])))
# df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['Label'], random_state=1)
# df_train, df_val = train_test_split(df_train, test_size=0.1, stratify=df_train['Label'], random_state=1)
#
# print((df_train['Label'].value_counts()))
# print((df_train['Label'].value_counts()/len(df_train)) * 100)
# print('\n=========================\n')
# print((df_val['Label'].value_counts()))
# print((df_val['Label'].value_counts()/len(df_val)) * 100)
# print('\n=========================\n')
# print((df_test['Label'].value_counts()))
# print((df_test['Label'].value_counts()/len(df_test)) * 100)


# split the English Translations into 50% development and 50% testing
df_en = pd.read_csv('sentences_ocr_corrected_discourse_profiling_en.csv')
df_en = df_en.dropna()
df_en = df_en[df_en['Label'] != 'Loaded_Language']
print(df_en['Label'].value_counts())
df_dev, df_test = train_test_split(df_en, test_size=0.5, stratify=df_en['Label'], random_state=42)

df_dev.to_csv('sentences_ocr_corrected_discourse_profiling_en_dev.csv', index=False)
df_test.to_csv('sentences_ocr_corrected_discourse_profiling_en_test.csv', index=False)

print('================================================')
# split the Arabic into 50% development and 50% testing
df_ar = pd.read_excel('sentences_ocr_corrected_discourse_profiling_ar.xlsx')
df_ar = df_ar.dropna()
df_ar = df_ar[df_ar['Label'] != 'Loaded_Language']
print(df_ar['Label'].value_counts())

df_dev, df_test = train_test_split(df_ar, test_size=0.5, stratify=df_ar['Label'], random_state=42)

df_dev.to_csv('sentences_ocr_corrected_discourse_profiling_ar_dev.xlsx', index=False)
df_test.to_csv('sentences_ocr_corrected_discourse_profiling_ar_test.xlsx', index=False)