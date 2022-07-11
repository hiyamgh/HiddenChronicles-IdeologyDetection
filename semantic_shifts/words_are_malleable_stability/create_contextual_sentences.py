import os
import pandas as pd


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

# w = 'المقاومه'
# path = 'C:/Users/96171/Downloads/'
# count = 0
# with open('2009_mukawama_beginning_withalta3reef.txt', 'w', encoding='utf-8') as fout:
#     # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#     with open(os.path.join(path, '2009.txt'), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             tokens = line.split(' ')
#             if w == tokens[0] or w in tokens[0]:
#                 # print(line)
#                 fout.write(line)
#                 count += 1
#         print('total count: {}/{}'.format(count, len(lines)))
#     f.close()
# fout.close()
#
#
# w = 'مقاومه'
# path = 'C:/Users/96171/Downloads/'
# count = 0
# with open('2009_mukawama_beginning_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#     # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#     with open(os.path.join(path, '2009.txt'), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             tokens = line.split(' ')
#             if w == tokens[0] or w in tokens[0] and 'المقاومه' not in tokens[0]:
#                 # print(line)
#                 fout.write(line)
#                 count += 1
#         print('total count: {}/{}'.format(count, len(lines)))
#     f.close()
# fout.close()

####################################################
# code when forcing to put "al-mukawama" at the beginning of the sentence:
# definition of mukawama: فالمقاومه الوطنيه الحقيقيه تستتد اليقاعده شعبيه موحده بينما عملت حماستقسيم الفلسطينيين وتمزيق وحدتهم واضعافهمبقوه السلاح والعنف والتسلط
# apparent denial: فالمقاومه الوطنيه الحقيقيه تستتد اليقاعده شعبيه موحده بينما عملت حماستقسيم الفلسطينيين وتمزيق وحدتهم واضعافهمبقوه السلاح والعنف والتسلط
# negative points of others (funny)
# المقاومه تخترع قضيه مزارعشبعا لتبرير الاحتفاظ بسلاحهاءانما الاحتلال اصراحتلالها والبقاء فيما والسيطرهالثروه المائيه الجوفيهارضهاء ياتي ظرف تتمالمؤامره المقاومه لاتهائهاالداخل ويستدرج العدوالحكومه اللبنانيه ليفلوضمافاوض ايار حول الترتيباتالامنيه واقتسام الميه
# فالمقاومه تنشا فراغ ولم تامعالدوله لحمايه جنوب لبتانء كانت امرا واضرورياء ورد فعل شعبيا طبيعيا العدوانالاسرائيلي

# positive points are referring to "al-mukawama" metaphorically
# فالمقاومه الوطنيه الحقيقيه تستتد اليقاعده شعبيه موحده بينما عملت حماستقسيم الفلسطينيين وتمزيق وحدتهم واضعافهمبقوه السلاح والعنف والتسلط
# لمقاومه العسكريه والسياسيه والاجتماعيهوالثقافيه والفكريه ماحصل لبنان عامنا ايمان يستطيعالشعوب تعاظم جبروته وتطورت ادارتهومهما بلغ الدعم المقدمتحدث عوض داعيا الي الوحده مواجههالعدوان
# لمقاومه العقليهتعني الوجه الجديد لعروبه مثقفه ترتضيتعددا للدول العربيه والتقافها نضاليا بعضهاانت تحارب اسراكيل بشكر اخر او غالبه
# المقاومه المدنيه تعتبر اليوم قضيه مشروعه مستمدهاعلان حقوق الانسان والمواطن والدستور الفرنسيوالتايات فرنسا للدوليه لحقوق الانسان والقانون الانساتيالي القبول بطموحاتها الثوريه وعتدما تتحدثالوزيره كلينتون هدف المفلوضات انما تتوقعارساء نظام اقليمي جديد تتعاون ايراندول الجوار وتقبل بالمشاركه نشر نفوذماءالتخلي سياسه المواجهه وتوزيع السلاح
#

# al-mukawama al falastiniya
# لمقاومه الفلسطينيه غزققائلا اننا نريد السلام والمقاومهمدفا حد ذاتماء قاذاانت ستؤدي الي تدمير الشعبتريدها
#

# found mukawama as a verb: ن حماستمارس عملها العسكري داخلفلسطين وتقوم بمقاومه الاحتلالداخل فلسطين
# found mukawama as a political party: المغزي ان سورياعشيه اتطلاق الحرب الاسرائيليه غزه انبد الانتقال الي المفلوضات المباشرهاسرائيل سعيا الي انجاز السلام تبادر الياعلان انتصار المقاومه وتجيير الاتتصاراليا فعلت غداه حرب تموز مقابلتخوين العرب الاخرين وذلك تقودتركيا حمله الاتصالات والمساعي حركهحماس” اجل وقف النار والتهذكهغزه بالتزامن حمله الامين العام حزب اللهالسيد حسن تصرالله مصر وحدهاءالساعين الي سلام اسراقيل سيكونجدوي قياسا تحققه المقاومهقال
# found mukawama in the context of Sayed Hasana Nasrallah (what he is saying): واعتبر ان مواقف الامين العامل حزب للله السيد حسن تصراللهالداعمه للمقاومه مشروعهومؤيده ان تناول الاتظمهالعربيه يؤدي الي مزيدالفرقهونوذ بوحده الصف الفلسطينيلبنان
# found mukawama as a verb in sports context: في المانيا يتصدر نادي بايرنالنوادي الالمانيه تحؤلت الاكثرللارياح اوروباء وعليه فان مقاومه النواديالمانيا تدعمها المناعه لمواجمهاي ازمات طاركه
#  found mukawama in context of shu3oob: تحقيق امدافه وان المقاومه جايث شوارع البللده انطلاقاالشعوب العربيه وفضبهامي ممثله الاسكوا تطالبه
# found mukawama as in mukawama falastiniya: انشطه التحرك والفعاليات سيواجه بمقاومه فلسطينيه وغزه اليوم تعيد تاكيد لاءات
# mukawama falastini yasser arafat: لتفاصيل كانت تنتظر لحظه التقاطعترقبت حماس انتهاء التهدكه لتطلق صارومامستوطنات الاحتلال ان تنتظر ردا بقساوه حدث املهتحرج رئيس السلطه الفلسطينيه محمود عباس وتزعزعالسلطه طردتها القطاع واتكرت دورها الي حددوس صورمؤسس المقاومه الفلسطينيه التاريخي ياسر عرقات واحراقها
# alot harakat hamas
# mukawama wataniya: لمقاومه الوطنيه الحقيقيه تستتد اليقاعده شعبيه موحده بينما عملت حماستقسيم الفلسطينيين وتمزيق وحدتهم واضعافهمبقوه السلاح والعنف والتسلط
# mukawama emil lahhoud: لقنطار ورحمهاستقبل الرئيس اميللحود امس منزله اليرزهالاسير المحرر سهير القنطارقال اللقاء انه اعربتقديره لهذا الرجل الكبيرادي دورا اساسياحمليه المقاومه والحفاظعروبه شعب لبنان ووحدته
#

# # mukawama - 1986 - nahar
# w = 'المقاومه'
# path = 'C:/Users/96171/Downloads/nahar_data'
# count = 0
# with open('1986_nahar_mukawama_beginning_withalta3reef.txt', 'w', encoding='utf-8') as fout:
#     # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#     with open(os.path.join(path, '1986.txt'), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             tokens = line.split(' ')
#             if w == tokens[0] or w in tokens[0]:
#                 # print(line)
#                 fout.write(line)
#                 count += 1
#         print('total count: {}/{}'.format(count, len(lines)))
#     f.close()
# fout.close()

# # mukawa - 1986 - assafir
# # w = 'المقاومه'
# # this is the beginning
# words = ['المقاومه', 'للمقاومه', 'مقاومه']
# years = ['1986', '2001', '2005', '2006']
# path = 'C:/Users/96171/Downloads/'
# archives = ['nahar', 'assafir']
# for year in years:
#     for archive in archives:
#         count = 0
#         with open('{}_{}_mukawama_beginning_withalta3reef.txt'.format(year, archive), 'w', encoding='utf-8') as fout:
#             # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#             with open(os.path.join(path + '{}_data/'.format(archive), '{}.txt'.format(year)), 'r', encoding='utf-8') as f:
#                 print('processing {} in {} archive'.format(year, archive))
#                 lines = f.readlines()
#                 for line in lines:
#                     tokens = line.split(' ')
#                     if tokens[0] in words or any([w in tokens[0] for w in words]):
#                         # print(line)
#                         fout.write(line)
#                         count += 1
#                 print('total count: {}/{}'.format(count, len(lines)))
#             f.close()
#         fout.close()

# this is the word anywhere
# words = ['المقاومه', 'للمقاومه', 'مقاومه']
years = ['1986', '2001', '2005', '2006']
path = 'C:/Users/96171/Downloads/'
archives = ['nahar', 'assafir']
keywords = ['سلاح', 'ارهاب', 'سوري', 'عراق', 'مسيحي', 'تحرير']
folders = ['weapon', 'terrorism', 'syria', 'iraq', 'christian', 'tahreer']
labels_political_discourse = ['active', 'euphimism', 'details', 'exaggeration', 'bragging', 'litote', 'repetition',
                                  'metaphor', 'he said', 'apparent denial', 'apparent concession', 'blame transfer',
                                  'other kinds', 'opinion', 'irony']
for i, keyword in enumerate(keywords):
    for year in years:
        for archive in archives:
            df = pd.DataFrame(columns=['Sentence', 'Technique', 'Span'] + labels_political_discourse)
            sentences = []
            print('processing for keyword {} in {}-{}'.format(keyword, year, archive))
            count = 0
            folder = 'contextual_analysis/{}/{}/'.format(folders[i], archive)
            mkdir(folder)
            with open(os.path.join(folder, '{}.txt'.format(year)), 'w', encoding='utf-8') as fout:
                # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
                with open(os.path.join(path + '{}_data/'.format(archive), '{}.txt'.format(year)), 'r', encoding='utf-8') as f:
                    # print('processing {} in {} archive'.format(year, archive))
                    lines = f.readlines()
                    for line in lines:
                        tokens = line.split(' ')
                        # if len(tokens) <= 20:
                        for t in tokens:
                            # if (t in words or any([w in t for w in words])) and (keyword in tokens or any([keyword in t for t in tokens])):
                            if keyword in tokens or any([keyword in t for t in tokens]):
                                # print(line)
                                fout.write(line)
                                sentences.append(line)
                                count += 1
                                break

                    print('total count: {}/{}, which is {}%%'.format(count, len(lines), (count/len(lines))*100))
                    df['Sentence'] = sentences
                    df.to_csv(os.path.join(folder, '{}.csv'.format(year)), index=False)
                    df.to_excel(os.path.join(folder, '{}.xlsx'.format(year)), index=False)
                f.close()
            fout.close()


# keywords = ['المناهضللارهاب', 'المميزهسوريا', 'الدعمتقدمه', 'الحاكمسوريا', 'الدبموقراطي', 'التحدياتتواجه']
# path = 'C:/Users/96171/Downloads/'
# count = 0
# words = ['المقاومه', 'للمقاومه', 'مقاومه']
# with open('test_keywords.txt', 'w', encoding='utf-8') as fout:
#     # with open('2009_mukawama_withoutalta3reef.txt', 'w', encoding='utf-8') as fout:
#     with open(os.path.join(path + 'nahar_data/', '2001.txt'), 'r', encoding='utf-8') as f:
#         lines = f.readlines()
#         for line in lines:
#             tokens = line.split(' ')
#             if any([w in tokens for w in words]) and any([k in tokens for k in keywords]):
#                 fout.write(line)
#                 count += 1
#         print('total count: {}/{}'.format(count, len(lines)))
#     f.close()
# fout.close()


#             # if word in tokens:
#             #     print(line)
#             #     fout.write(line)
#             #     count += 1
#             # else:
#             #     for t in tokens:
#             #         if word in t and 'المقاومه' not in t and 'للمقاومه' not in t: # if its a subword of a certain token - because of OCR errors
#             #             print(line)
#             #             fout.write(line)
#             #             count += 1
#             #             break


# realized that al-mukawama and ll-mukawama resemble the political party
# realized that mukawama alone does not resemble political party
# mukawama alone:
# risalat mukawama line 1
# kiwa mukawama musalla7a line 2
# mukawamat sha3b line 3
# mukawamat 3aduw line 4
# mukawamt al shark al awsat line 12
# mukawamat alsharr line 71
# mukawat al mar2a line 72
# mukawama kabira llta8yeer wal isla7 line 96
# mukawat aljoo3 wal jahel
# mukawa sa3eed idologia w sa3eed tatbeeq wa ittisa3 line 201
# mukawama roo7iya ra3awiyya tarbawiyya  line 263
#