from googletrans import Translator

# # nationality = ['Israeli', 'Palestinian', 'Israelis', 'Jewish', 'Jews']
# nationality = [
# 'إسرائيلي',
# 'فلسطيني',
# 'الإسرائيليين',
# 'يهودي',
# 'يهود',
# ]
#
# # ===========================================================================================
#
# # location_palestine = ['West Bank', 'Hebron', 'Gaza', 'Ramallah', 'Beit', 'Beit', 'Issawiya', 'Aqsa', 'Shuafat']
# location_palestine = [
# 'الضفة الغربية',
# 'الخليل',
# 'غزة',
# 'رام الله',
# 'بيت',
# 'بيت',
# 'العيسوية',
# 'الأقصى',
# 'شعفاط',
# ]
#
#
# # location_israel = ['Israel', 'Jerusalem', 'Beit', 'Tel Aviv']
# location_israel = [
# 'إسرائيل',
# 'بيت المقدس',
# 'بيت',
# 'تل أبيب',
# ]
#
# # ===========================================================================================
#
# # leadership_palestine = ['Abbas']
# leadership_palestine = ['عباس']
#
# # leadership_israel = ['Biden', 'Netanyahu', 'Eisenkot', 'Yaalon']
# leadership_israel = [
# 'بايدن',
# 'نتنياهو',
# 'ايزنكوت',
# 'يعالون',
# ]
#
#
# # ===========================================================================================
#
#
#
# # ===========================================================================================
# # military = ['soldiers', 'military', 'forces', 'assailants', 'police', 'militant', 'security', 'authorities']
# military = [
# 'جنود',
# 'الجيش',
# 'القوات',
# 'المهاجمين',
# 'شرطة',
# 'مناضل',
# 'الأمان',
# 'السلطات',
# ]
#
# citizens = ['residents', 'girl', 'youths']
#
# # negative = ['Washington', 'America', 'United States', 'European Union', 'Emergency']
# negative = [
# 'واشنطن',
# 'أمريكا',
# 'الولايات المتحدة الأمريكية',
# 'الإتحاد الأوربي',
# 'حالة طوارئ',
# ]
#
# # https://interactive.aljazeera.com/aje/palestineremix/phone/glossary_main.html
# # war_key_terms = ['administrative', 'detention', 'borders', 'bypass', 'road','closure','collective',
# #                  'punishment', 'ethnic', 'cleansing', 'green', 'line', 'intifada', 'occuppied', 'palestinian',
# #                  'territories', 'authority', 'refugee', 'settlement', 'separation', 'wall',
# #                  'zionism']
# war_key_terms = [
# 'إداري',
# 'احتجاز',
# 'الحدود',
# 'تجاوز',
# 'طريق',
# 'إغلاق',
# 'جماعي',
# 'عقاب',
# 'عرقي',
# 'تطهير',
# 'أخضر',
# 'خط',
# 'الانتفاضة',
# 'مسدود',
# 'فلسطيني',
# 'إقليم',
# 'السلطة',
# 'لاجئ',
# 'مستوطنة',
# 'انفصال',
# 'حائط',
# 'صهيونية',
# ]
#
# # =========================================================================================================
# key_terms2 = ['Balfour', 'Declaration', 'Fatah', 'Flotilla', 'Gaza', 'Genava', 'Accord', 'Golan',
#               'Heights', 'Green', 'Line', 'Hamas', 'Haram', 'Hezbollah', 'Intifada', 'Israel', 'Defense', 'Forces',
#               'Knesset', 'Nakba', 'Oslo', 'Accords', 'Palestinian', 'Authority', 'Palestine', 'Liberation', 'organization',
#               'National', 'warfare', 'battle', 'conflict', 'fight', 'struggle', 'combat']
#
# # violence = ['killed', 'attacks', 'violence', 'stabbing', 'shot', 'Wounded', 'clashes', 'occupation']
# violence = [
# 'قتل',
# 'الهجمات',
# 'عنف',
# 'طعن',
# 'اطلاق النار',
# 'جرحى',
# 'اشتباكات',
# 'الاحتلال',
# ]
#
# # amity = ['peace', 'truce', 'unity', 'amity']
# amity = [
# 'سلام',
# 'هدنة',
# 'وحدة',
# 'الصداقة',
# ]

location_palestine = ['West Bank', 'Hebron', 'Gaza', 'Ramallah', 'Issawiyah', 'Aqsa', 'Shuafat']
location_israel = ['Israel', 'Jerusalem', 'Tel Aviv']

if __name__ == '__main__':
    translator = Translator()
    print('[')
    # for term in war_key_terms:
    # for term in negative:
    # for term in military:
    # for term in violence:
    # for term in amity:
    # for term in leadership_israel:
    # for term in location_palestine:
    # for term in location_israel:
    for term in location_israel:
        print('\'{}\','.format(translator.translate(term, dest='ar').text))
    print(']')