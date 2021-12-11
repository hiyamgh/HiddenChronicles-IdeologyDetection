import mwparserfromhell
import pywikibot

enwp = pywikibot.Site('ar', 'wikipedia')
page = pywikibot.Page(enwp, 'بيار الجميل')
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