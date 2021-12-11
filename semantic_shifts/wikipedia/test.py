import mwparserfromhell
import pywikibot

enwp = pywikibot.Site('en', 'wikipedia')
page = pywikibot.Page(enwp, 'Arab Liberation Front')
wikitext = page.get()
wikicode = mwparserfromhell.parse(wikitext)

templates = wikicode.filter_templates()
for t in templates:
    print(t)
    print('-------------------------------------------')

# from pprint import pprint
# import pywikibot
# site = pywikibot.Site('wikipedia:en')  # or pywikibot.Site('en', 'wikipedia') for older Releases
# page = pywikibot.Page(site, 'Khalil_Gebran')
# all_templates = page.raw_extracted_templates()
# for tmpl, params in all_templates:
#     if tmpl == 'Infobox film':
#         pprint(params)
# # for tmpl, params in all_templates:
# #     if tmpl.title(with_ns=False) == 'Infobox film':
# #         pprint(tmpl)

# import wikipedia
# ny = wikipedia.page("Nabih Berry")
# print(ny.links)
# u'1790 United States Census'

# wikipedia.set_lang("fr")
# wikipedia.summary("Facebook", sentences=1)
# Facebook est un service de réseautage social en lign

# from: https://stackoverflow.com/questions/25122445/python-wikipedia-scraping-getting-the-links-to-same-page-in-other-languages

import urllib3
from bs4 import BeautifulSoup

http = urllib3.PoolManager()
url = 'http://en.wikipedia.org/wiki/Musa al-Sadr'
response = http.request('GET', url)

# get languages and links
soup = BeautifulSoup(response.data)
links = [(el.get('lang'), el.get('href')) for el in soup.select('li.interlanguage-link > a')]

for language, link in links:
    # url = 'http:' + link
    response = http.request('GET', link)
    soup = BeautifulSoup(response.data)
    print(language, soup.title.text, link)