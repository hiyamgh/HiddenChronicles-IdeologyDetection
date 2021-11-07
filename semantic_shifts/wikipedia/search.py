import wikipedia
import pickle
import pandas as pd
import requests
import json

S = requests.Session()

res = wikipedia.search("Lebanon War", results=4, suggestion=False)
all_categories = {}
all_categories_set = set()
count = 1
df = pd.DataFrame()
for a in res:
    article_page = wikipedia.page(a)
    print("{}: article title: {}, url: {}\nCategories: {}".format(count, article_page.title, article_page.url, article_page.categories))
    all_categories[article_page.title] = article_page.categories
    for c in article_page.categories:
        all_categories_set.add(c)
    count += 1

with open('categories.pickle', 'wb') as handle:
    pickle.dump(all_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('All Categories Set:\n{}\n'.format(all_categories_set))

# define the requests session and the url of media wiki
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"

max_times = 5
times = 0
# now, for each category encountered, get all its sub-categories
for c in all_categories_set:
    print('Processing Category: {}'.format(c))
    success = 0
    PARAMS = {
        "action": "query",
        "cmtitle": "Category:{}".format(c),
        "cmlimit": "max",  # get all its subcategories
        "cmtype": "subcat",
        "list": "categorymembers",
        "format": "json",
    }

    while times < max_times:
        try:
            R = S.get(url=URL, params=PARAMS)
            DATA = R.json()
            print('number of sub-categories: {}'.format(len(DATA["query"]["categorymembers"])))
            with open('{}-subcategories.json'.format(c), 'w', encoding='utf-8') as f:
                json.dump(DATA, f, ensure_ascii=False, indent=4)
            success = 1
            break
        except requests.exceptions.ConnectionError:
            print("Retrying \'{w}\' due to connection error".format(w=c))
            times += 1
    if success == 1:
        success = 0
    else:
        print("\'{w}\' failed too many times ({t}) times. " + "Moving on".format(w=c, t=times))
