import os

dirs = ['translationsv2/van_dijk_speeches/', 'translationsv2/van_dijk_contexts/',
            'translationsv2/propaganda/', 'translationsv2/argumentation/']

groups = ['Palestinian Resistance South Lebanon/', 'Sabra and Shatila Massacre/']
groups_van_djik = ['Palestinian Resistance South Lebanon_cleaned/', 'Sabra and Shatila Massacre_cleaned/']

annotations = ["VDS", "VDC", "PTC", "ARG"]
for i, dir in enumerate(dirs):
    if annotations[i] != "VDC":
        for group in groups:
            path = os.path.join(dir, group)
            added = "../translate_corpora/annotations_doccano/"
            for file in os.listdir(path):
                if '.xlsx' in file:
                    lang = file.split('_')[2][:-5]
                    if 'Palestinian' in group:
                        print("\"corp_PRST_{}_{}\": \"{}\",".format(lang, annotations[i], added + path + file))
                    else:
                        print("\"corp_SSM_{}_{}\": \"{}\",".format(lang, annotations[i], added + path + file))
    else:
        for group in groups_van_djik:
            path = os.path.join(dir, group)
            added = "../translate_corpora/annotations_doccano/"
            for file in os.listdir(path):
                if '.xlsx' in file:
                    lang = file.split('_')[2][:-5]
                    if 'Palestinian' in group:
                        print("\"corp_PRST_{}_{}\": \"{}\",".format(lang, annotations[i], added + path + file))
                    else:
                        print("\"corp_SSM_{}_{}\": \"{}\",".format(lang, annotations[i], added + path + file))