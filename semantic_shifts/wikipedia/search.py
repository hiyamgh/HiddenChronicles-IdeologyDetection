import wikipedia
import pickle
import pandas as pd
import requests
import json

S = requests.Session()

res = wikipedia.search("Lebanon War", results=100, suggestion=False)
all_categories = {}
all_categories_set = set()
count = 1
for a in res:
    try:
        article_page = wikipedia.page(a)
        print("{}: article title: {}, url: {}\nCategories: {}".format(count, article_page.title, article_page.url, article_page.categories))
        all_categories[article_page.title] = article_page.categories
        for c in article_page.categories:
            if 'Articles containing' not in c:
                all_categories_set.add(c)
    except wikipedia.exceptions.PageError:
        # if a "PageError" was raised, ignore it and continue to next link
        continue
    except wikipedia.DisambiguationError as e:
        continue
    count += 1

with open('categories.pickle', 'wb') as handle:
    pickle.dump(all_categories, handle, protocol=pickle.HIGHEST_PROTOCOL)

print('All Categories Set:\n{}\n'.format(all_categories_set))

# define the requests session and the url of media wiki
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"

max_times = 5
times = 0
cats_with_no_subcats = []
len_mod_300 = 1
df = pd.DataFrame(columns=['category', 'sub-category'])
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
            length = len(DATA["query"]["categorymembers"])
            print('number of sub-categories: {}'.format(length))
            if length > 0:
                subcats = DATA["query"]["categorymembers"]
                count = 0
                for sub in subcats:
                    if count == 0:
                        df = df.append({
                            'category': c,
                            'sub-category': sub['title']
                        }, ignore_index=True)
                    else:
                        df = df.append({
                            'category': '',
                            'sub-category': sub['title']
                        }, ignore_index=True)

                    count += 1

                # with open('{}-subcategories.json'.format(c), 'w', encoding='utf-8') as f:
                #     json.dump(DATA, f, ensure_ascii=False, indent=4)
            else:
                df = df.append({
                    'category': c,
                    'sub-category': ''
                }, ignore_index=True)

                print('will not save \'{}\' because it has no subcategories'.format(c))
                cats_with_no_subcats.append(c)
                pass

            if len(df) % 300 == 0:
                df.to_csv('categories-batch-{}.csv'.format(len_mod_300), index=False)
                len_mod_300 += 1
                df = pd.DataFrame(columns=['category', 'sub-category'])
            success = 1
            break
        except requests.exceptions.ConnectionError:
            print("Retrying \'{w}\' due to connection error".format(w=c))
            times += 1
    if success == 1:
        success = 0
    else:
        print("\'{w}\' failed too many times ({t}) times. " + "Moving on".format(w=c, t=times))

df.to_csv('categories-batch-{}.csv'.format(len_mod_300), index=False)

# save list of all categories that have no subcategories:
with open("all_cats_with_no_subcats.txt", "wb") as fp:
    pickle.dump(cats_with_no_subcats, fp)