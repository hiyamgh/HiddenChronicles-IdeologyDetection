import os

dirs = ['translationsv2/van_dijk_speeches/', 'translationsv2/van_dijk_contexts/',
            'translationsv2/propaganda/', 'translationsv2/argumentation/']

groups = ['Palestinian Resistance South Lebanon/', 'Sabra and Shatila Massacre/']

annotations = ["VDS", "VDC", "PTC", "ARG"]
for i, dir in enumerate(dirs):
    for group in groups:
        path = os.path.join(dir, group)
        for file in os.listdir(path):
            if '.xlsx' in file:
                lang = file.split('_')[2][:-5]
                if 'Palestinian' in group:
                    print("\"corp_PRST_{}_{}\": \"{}\",".format(lang, annotations[i], path + file))
                else:
                    print("\"corp_SSM_{}_{}\": \"{}\",".format(lang, annotations[i], path + file))