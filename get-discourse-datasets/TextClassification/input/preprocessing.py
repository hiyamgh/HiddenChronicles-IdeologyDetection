import pandas as pd
import os
from sklearn.model_selection import train_test_split
from googletrans import Translator
import time
import re


def get_class_percentages(df, label_col, mode):
    percentages = (df[label_col].value_counts() / len(df)) * 100
    print('\npercentages in {} data:\n'.format(mode))
    print(percentages)
    print('shape of df: {}'.format(df.shape))


def translate_dataset(df, text_col):
    print('len of df: {}'.format(len(df)))
    df[text_col + '_ar'] = df.apply(lambda x: translator.translate(x[text_col], dest='ar').text, axis=1)
    return df


def get_cleaned(text):
    s = re.sub(r'\s*[A-Za-z]+\b', '', text)
    s = re.sub(" \d+", " ", s)
    s = re.sub(r'\b\d+(?:\.\d+)?\s+', '', s)
    s = re.sub('[()]', '', s) # remove paranthesis
    s = re.sub(r'(?:\-|\s|(\d+))(?=[^><]*?<\/u>)', '', s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub(' +', ' ', s)  # remove multiple spaces
    return s



def clean_data(df, text_col):

    def get_cleaned(text):
        s = re.sub(r'\s*[A-Za-z]+\b', '', text)
        s = re.sub(" \d+", " ", s)
        s = re.sub(r'\b\d+(?:\.\d+)?\s+', '', s)
        s = re.sub(' +', ' ', s) # remove multiple spaces


if __name__ == '__main__':
    # text = 'سيتم تضمين أسهم الاتصالات الثلاثة - Verizon Communications Inc (VZ.N) و AT&T Inc (T.N) و CenturyLink Inc (CTL.N) - قطاع خدمات الاتصالات S&P 500 الجديد يوم الاثنين ، إلى جانب كبار الشخصيات بما في ذلك Netflix Inc (NFLX.O) و Alphabet Inc (GOOGL.O) و Facebook Inc (FB.O).'
    # print(get_cleaned(text))

    if os.path.isfile('df_train.xlsx'):
        df_train = pd.read_excel('df_train.xlsx',  index_col=False)
        df_dev = pd.read_excel('df_dev.xlsx', index_col=False)
        df_test = pd.read_excel('df_test.xlsx', index_col=False)

        # get the number of instances/percentage for the class NA
        get_class_percentages(df_train, label_col='Label', mode='train')
        print('NA class in train: {}'.format((df_train['Label'].isna().sum() / len(df_train)) * 100))

        get_class_percentages(df_dev, label_col='Label', mode='dev')
        print('NA class in dev :{}'.format((df_dev['Label'].isna().sum() / len(df_dev)) * 100))

        get_class_percentages(df_test, label_col='Label', mode='test')
        print('NA class in test :{}'.format((df_test['Label'].isna().sum() / len(df_test)) * 100))

        # replace all Nans in with a 'No_class' class
        df_train['Label'] = df_train['Label'].fillna('No_class')
        df_dev['Label'] = df_dev['Label'].fillna('No_class')
        df_test['Label'] = df_test['Label'].fillna('No_class')

        print('\n=============================== After replacing NA with No_class class ===============================')
        get_class_percentages(df_train, label_col='Label', mode='train')
        get_class_percentages(df_dev, label_col='Label', mode='dev')
        get_class_percentages(df_test, label_col='Label', mode='test')

    else:
        df = pd.read_csv('NewsDiscourse_politicaldiscourse.csv')
        text_column = 'Sentence'
        label_column = 'Label'
        translator = Translator()

        df = df[[text_column, label_column]]
        t1 = time.time()
        df = translate_dataset(df, text_col=text_column)
        t2 = time.time()
        print('time taken to translate: {} mins'.format((t2-t1)/60))

        # split dataset into 80% training, 10% development, 10% testing
        df_train, df_test = train_test_split(df, test_size=0.1, random_state=42, stratify=list(df[label_column]))
        df_train, df_dev = train_test_split(df_train, test_size=0.1, random_state=42, stratify=list(df_train[label_column]))

        # print class percentages in each dataset
        get_class_percentages(df=df_train, label_col=label_column, mode='train')
        get_class_percentages(df=df_dev, label_col=label_column, mode='dev')
        get_class_percentages(df=df_test, label_col=label_column, mode='test')

        df_train.to_excel('df_train.xlsx', index=False, encoding='utf-8-sig')
        df_dev.to_excel('df_dev.xlsx', index=False, encoding='utf-8-sig')
        df_test.to_excel('df_test.xlsx', index=False, encoding='utf-8-sig')

