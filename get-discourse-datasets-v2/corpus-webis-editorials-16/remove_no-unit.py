import pandas as pd
import os


def mkdir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


if __name__ == '__main__':
    directory = "translationsv2/"
    save_dir = "translationsv2_cleaned/"
    for file in os.listdir(directory):
        df = pd.read_excel(os.path.join(directory, file))
        df = df[df.Label != "no-unit"]
        mkdir(save_dir)
        df.to_excel(os.path.join(save_dir, file))

