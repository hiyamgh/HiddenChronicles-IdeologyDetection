# //*[@id="mw-content-text"]/div[1]/center[1]/table[2]
import requests
from bs4 import BeautifulSoup
import pandas as pd
import os


def get_sum(nb):
    nb = nb.replace("(", "")
    nb = nb.replace(")", "")
    nb1, nb2 = nb.split(" ")[0], nb.split(" ")[1]
    print("sum of {} and {} is: {}".format(nb1, nb2, int(nb1) + int(nb2)))
    return int(nb1) + int(nb2)


def mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


if __name__ == '__main__':
    url = 'https://en.wikipedia.org/wiki/Israeli%E2%80%93Palestinian_conflict'
    s = requests.session()
    response = s.get(url, timeout=10)
    print(response)

    # define beautiful soup object
    soup = BeautifulSoup(response.content, 'html.parser')

    # get the wikipedia table of ineterst
    wiki_table = soup.find('table', {'class': 'wikitable'})

    # get the rows
    rows = wiki_table.findAll("tr")

    # build list of lists for building the data frame
    lst_data = []
    for row in rows[2:-2]:
        data = [d.text.rstrip() for d in row.find_all('td')]
        data.extend([d.text.rstrip() for d in row.find_all('th')])
        lst_data.append(data)

    # build the data frame
    df = pd.DataFrame(columns=['Year', 'Palestinians', 'Israelis'])
    df['Year'] = [sub_list[2] for sub_list in lst_data]
    df['Palestinians'] = [sub_list[0] for sub_list in lst_data]
    df['Israelis'] = [sub_list[1] for sub_list in lst_data]

    for idx, row in df.iterrows():
        df.loc[idx, 'Palestinians'] = get_sum(row['Palestinians'])
        df.loc[idx, 'Israelis'] = get_sum(row['Israelis'])

    df = df.sort_values(by="Year", ascending=True)
    # out_dir = "casualties"
    # mkdir(out_dir)
    df.to_csv("casualties_1988_2011.csv")
