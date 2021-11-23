import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    "action": "query",
    "format": "json",
    "list": "embeddedin",
    "eititle": "Computer",
    "eilimit": "20"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["embeddedin"]

for p in PAGES:
    print(p["title"])

# #!/usr/bin/python3
#
# """
#     python/get_iwlinks.py
#
#     MediaWiki API Demos
#     Demo of `Iwlinks` module: Get the interwiki links from a given page.
#
#     MIT License
# """
#
# import requests
#
# S = requests.Session()
#
# URL = "https://en.wikipedia.org/w/api.php"
#
# PARAMS = {
#     "action": "query",
#     "format": "json",
#     # "prop": "iwlinks",
#     # "prop":   "extlinks",
#     "prop":   "links",
#     "titles": "Albert Einstein"
# }
#
# R = S.get(url=URL, params=PARAMS)
# DATA = R.json()
#
# PAGES = DATA["query"]["pages"]
#
# for k, v in PAGES.items():
#     # print(v["iwlinks"])
#     # print(v["extlinks"])
#     print(v["links"])