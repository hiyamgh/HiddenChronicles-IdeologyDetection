import pandas as pd
import os
from googletrans import Translator
import re


def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


translator = Translator()
all_keywords = []

# getting ethnicities
ethn = pd.read_csv('from_DrFatima/ethnicities_races.csv')
# drop the 4th column it only has a note
ethn = ethn.drop(['Unnamed: 3'], axis=1)
print('len before: {}'.format(len(ethn)))
# filter the dataset, get only rows that contain '*'
ethn = ethn[ethn['Unnamed: 2'].str.strip() == '*'].reset_index(drop=True)
print('len after: {}'.format(len(ethn)))
# save
dirc = 'from_DrFatima_cleaned/'
mkdir(dirc)
ethn.to_csv(os.path.join(dirc, 'ethnicities_races.csv'), encoding='utf-8-sig', index=False)
# save arabic ethnicities as txt file
ethnicities_ar = list(ethn['ar'])
ethnicities_en = list(ethn['en'])
with open(os.path.join(dirc, 'ethnicities.txt'), 'w', encoding='utf-8') as f:
    for i, e in enumerate(ethnicities_ar):
        if str(e) == 'nan':
            e = 'ايراني'
        if str(e) == 'الأوصياء':
            e = 'حراس الارز'
        if str(e) == 'IAF':
            continue
        f.write(e + '\n')
        all_keywords.append(e) # add to the list of all keywords (ideologiess + ethnicities + politicians + political parties)
f.close()

# political_parties
political_parties = pd.read_csv('from_DrFatima/political_parties_ar.csv')
# get ideologies
allideologies = []
ideologies = list(political_parties['الإيديولوجيا'])
for ideo in ideologies:
    splitted = ideo.split(',')
    for id in splitted:
        id = re.sub('-', '', id)
        allideologies.append(id)
allideologies = list(set(allideologies))
allideologies = [id for id in allideologies if id.strip() != '']

# get extra ideologies that Dr. Fatima pointed out
ideologies_extra = pd.read_excel('from_DrFatima/ideologies.xlsx', header=None)
ideologies_en = list(ideologies_extra[0])
ideologies_ar = []
en2ar = {
    'Waliayat Al Faqih': 'ولايه الفقيه',
    'Mullas': 'الملا'
}

for id_en in ideologies_en:
    if id_en in en2ar:
        id_ar = en2ar[id_en]
    else:
        id_ar = translator.translate(id_en, src='en', dest='ar').text
    print('{} => {}'.format(id_en, id_ar))
    # id_ar = re.sub('-', '', id_ar)
    ideologies_ar.append(id_ar)

all_together = allideologies + ideologies_ar
unwanted_words = [
    'ليتوانيا',
    'أغسطس',
    'عموم سوريا',
     'سنف',
    'شرطة (علامة)',
    'ابن',
    'المصالح العربية الإسرائيلية',
    'قائمة عادية',
    'هناك حاجة إلى صفحة مناهضة الشيوعية',
    'شيما',
    'إنذار بالدفع',
    'قائمة قابلة للطي',
    'حل واحد',
    'القومية الفلسطينية معاداة الصهيونية',
    'Antirevisionism الليبرتارية الاشتراكية Anticapitalism',
]

all_together = [id for id in all_together if id not in unwanted_words]
ideologies_df = pd.DataFrame()
ideologies_df['الإيديولوجيا'] = all_together
mkdir(dirc)
ideologies_df.to_csv(os.path.join(dirc, 'ideologies_ar.csv'), encoding='utf-8-sig', index=False)
# save arabic ideologies as txt file
ideo_ar = list(ideologies_df['الإيديولوجيا'])
with open(os.path.join(dirc, 'ideologies.txt'), 'w', encoding='utf-8') as f:
    for e in ideo_ar:
        f.write(e + '\n')
        all_keywords.append(e) # add to the list of all keywords (ideologiess + ethnicities + politicians + political parties)
f.close()

# filter the dataset, get only rows that contain '*'
print('len before: {}'.format(len(political_parties)))
political_parties = political_parties[political_parties['Unnamed: 4'].str.strip() == '*'].reset_index(drop=True)
print('len after: {}'.format(len(political_parties)))
mkdir(dirc)
political_parties.to_csv(os.path.join(dirc, 'political_parties_ar.csv'), encoding='utf-8-sig', index=False)
# save arabic political parties as txt file
political_parties_ar = list(political_parties['الاسم'])
with open(os.path.join(dirc, 'political_parties.txt'), 'w', encoding='utf-8') as f:
    for e in political_parties_ar:
        f.write(e + '\n')
        all_keywords.append(e) # add to the list of all keywords (ideologiess + ethnicities + politicians + political parties)
f.close()

# politicians
politicians = pd.read_csv('from_DrFatima/politicians_ar.csv')
print('len before: {}'.format(len(politicians)))
politicians = politicians[politicians['Unnamed: 4'].str.strip() == '*'].reset_index(drop=True)
print('len after: {}'.format(len(politicians)))
politicians.to_csv(os.path.join(dirc, 'politicians_ar.csv'), encoding='utf-8-sig', index=False)
# save arabic politicians as txt file
politicians_ar = list(politicians['الاسم'])
with open(os.path.join(dirc, 'politicians.txt'), 'w', encoding='utf-8') as f:
    for e in politicians_ar:
        f.write(e + '\n')
        all_keywords.append(e) # add to the list of all keywords (ideologiess + ethnicities + politicians + political parties)
f.close()

with open(os.path.join(dirc, 'all_keywords.txt'), 'w', encoding='utf-8') as f:
    for k in all_keywords:
        f.write(k + '\n')
f.close()