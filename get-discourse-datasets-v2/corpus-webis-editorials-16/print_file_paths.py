import os
if __name__ == '__main__':
    # annotations = ["VDS", "VDC", "PTC", "ARG"]
    dir = 'translationsv2/'
    path_en = 'sentences_annotations.xlsx'
    for file in os.listdir(dir):
        if '.xlsx' in file:
            lang = file.split('_')[2][:-5]
            path = os.path.join(dir, file)
            print("\"ARG_{}\": \"{}\",".format(lang, path))

    print("\"ARG_en\": \"{}\",".format(path_en))