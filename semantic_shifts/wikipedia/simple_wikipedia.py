'''
- describe each timeline
- scrape for article around keywords above
- follow procedure of hyperlink-category
- its okay to generate list of keywords manually
'''

import wikipedia
import requests

session = requests.Session()

url = "https://en.wikipedia.org/w/api.php"
params = {
    "action": "query",
    "format": "json",
    "titles": "Lebanese Civil War",
    "prop": "links",
    "pllimit": "max"
}

response = session.get(url=url, params=params)
data = response.json()
pages = data["query"]["pages"]

pg_count = 1
page_titles = []

print("Page %d" % pg_count)
for key, val in pages.items():
    for link in val["links"]:
        print(link["title"])
        page_titles.append(link["title"])

while "continue" in data:
    plcontinue = data["continue"]["plcontinue"]
    params["plcontinue"] = plcontinue

    response = session.get(url=url, params=params)
    data = response.json()
    pages = data["query"]["pages"]

    pg_count += 1

    print("\nPage %d" % pg_count)
    for key, val in pages.items():
        for link in val["links"]:
            print(link["title"])
            page_titles.append(link["title"])

print("%d titles found." % len(page_titles))

# lbc = wikipedia.page("Lebanese Civil War")
# links = lbc.links
# for link in links:
#     print(link)