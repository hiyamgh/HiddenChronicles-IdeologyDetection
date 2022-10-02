import pandas as pd
from sklearn.model_selection import train_test_split
from googletrans import Translator

df = pd.read_csv('NewsDiscourse_politicaldiscourse.csv')
df = df[['Sentence', 'Label']]
print('before dropping nans: df.shape: {}'.format(df.shape))
df = df.dropna()
print('after dropping nans: df.shape: {}'.format(df.shape))
df.to_csv('NewsDiscourse_politicaldiscourse_nonans.csv', index=False)

print(df['Label'].value_counts())
print((df['Label'].value_counts()/len(df))*100)
print('/////////////////////////////////////////////////////////////////')
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Label'])
df_train.to_csv('df_train_en.csv', index=False)
df_val.to_csv('df_val_en.csv', index=False)

# Make an Arabic version
translator = Translator()

# translate the training data
sentences_en = df_train['Sentence']
sentences_ar = []
for sent in sentences_en:
    sentences_ar.append(translator.translate(sent, src='en', dest='ar').text)

df_train_ar = pd.DataFrame()
df_train_ar['Sentence_ar'] = sentences_ar
df_train_ar['Label'] = df_train['Label']
df_train_ar.to_excel('df_train_ar.xlsx')

# translate the validation data
sentences_en = df_val['Sentence']
sentences_ar = []
for sent in sentences_en:
    sentences_ar.append(translator.translate(sent, src='en', dest='ar').text)

df_val_ar = pd.DataFrame()
df_val_ar['Sentence_ar'] = sentences_ar
df_val_ar['Label'] = df_val['Label']
df_val_ar.to_excel('df_val_ar.xlsx')

#
# print(df_train['Label'].value_counts())
# print((df_train['Label'].value_counts()/len(df_train)) * 100)
# print('========================================\n\n')
# print(df_val['Label'].value_counts())
# print((df_val['Label'].value_counts()/len(df_val)) * 100)


# C:\Users\96171\AppData\Local\Programs\Python\Python36\python.exe C:/Users/96171/Desktop/political_discourse_mining_hiyam/get-discourse-datasets/TextClassification/input/Discourse_Profiling/preprocessing_new.py
# Distant_Evaluation                   12981
# Cause_General                        11680
# Distant_Expectations_Consequences     6271
# Main                                  4675
# Distant_Historical                    3432
# Cause_Specific                        3002
# Distant_Anecdotal                     1373
# Main_Consequence                       773
# Name: Label, dtype: int64
# Distant_Evaluation                   29.377419
# Cause_General                        26.433114
# Distant_Expectations_Consequences    14.191957
# Main                                 10.580035
# Distant_Historical                    7.766990
# Cause_Specific                        6.793853
# Distant_Anecdotal                     3.107249
# Main_Consequence                      1.749383
# Name: Label, dtype: float64
# /////////////////////////////////////////////////////////////////
# Distant_Evaluation                   10385
# Cause_General                         9344
# Distant_Expectations_Consequences     5017
# Main                                  3740
# Distant_Historical                    2745
# Cause_Specific                        2402
# Distant_Anecdotal                     1098
# Main_Consequence                       618
# Name: Label, dtype: int64
# Distant_Evaluation                   29.378483
# Cause_General                        26.433562
# Distant_Expectations_Consequences    14.192764
# Main                                 10.580214
# Distant_Historical                    7.765425
# Cause_Specific                        6.795100
# Distant_Anecdotal                     3.106170
# Main_Consequence                      1.748281
# Name: Label, dtype: float64
# ========================================
#
#
# Distant_Evaluation                   2596
# Cause_General                        2336
# Distant_Expectations_Consequences    1254
# Main                                  935
# Distant_Historical                    687
# Cause_Specific                        600
# Distant_Anecdotal                     275
# Main_Consequence                      155
# Name: Label, dtype: int64
# Distant_Evaluation                   29.373161
# Cause_General                        26.431319
# Distant_Expectations_Consequences    14.188730
# Main                                 10.579317
# Distant_Historical                    7.773252
# Cause_Specific                        6.788866
# Distant_Anecdotal                     3.111564
# Main_Consequence                      1.753790
# Name: Label, dtype: float64
#
# Process finished with exit code 0