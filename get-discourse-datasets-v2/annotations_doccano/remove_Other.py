import pandas as pd
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    directories = [
        "translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon/",
        "translationsv2/van_dijk_contexts/Sabra and Shatila Massacre/",
    ]

    directories_cleaned = [
        "translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/",
        "translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/",
    ]

    for i, direc in enumerate(directories):
        for file in os.listdir(direc):
            df = pd.read_excel(os.path.join(direc, file))
            df = df[df.label != "Other"]
            mkdir(directories_cleaned[i])
            df.to_excel(os.path.join(directories_cleaned[i], file))


