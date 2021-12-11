import pandas as pd
from googletrans import Translator


from deep_translator import GoogleTranslator
to_translate = 'I want to translate this text'
translated = GoogleTranslator(source='auto', target='ar').translate(to_translate)
print(translated)
# outpout -> Ich möchte diesen Text übe


#
# translator = Translator()

df = pd.read_csv('datasets/poltical_parties.csv')
ideologies = df['Ideology']
for i in ideologies:
    # i = str(i)
    # ll = i.split(',') if ',' in i else i
    # if isinstance(ll, list):
    #     translations = translator.translate(ll, src='en', dest='ar')
    #     for translation in translations:
    #         print(translation.origin, ' -> ', translation.text)
    # else:
    if i!= i:
        continue
    translated = GoogleTranslator(source='auto', target='ar').translate(i)
    print(i, ' -> ', translated)
    print('---------------------------------')

#
# translation = translator.translate("Hola Mundo", dest="ar")
# print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")

# translations = translator.translate(['The quick brown fox', 'jumps over', 'the lazy dog'], dest='ko')
# for translation in translations:
#     print(translation.origin, ' -> ', translation.text)