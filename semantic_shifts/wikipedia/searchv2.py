import wikipedia
import pickle
import pandas as pd
import requests
import json

S = requests.Session()
article_page1 = wikipedia.page("Sabra_and_Shatila_massacre")
article_page2 = wikipedia.page("Synth-pop")

df_page1 = pd.DataFrame(columns=['category', 'relevant'])
df_page2 = pd.DataFrame(columns=['category', 'relevant'])

df_page1['category'] = [c for c in article_page1.categories]
df_page2['category'] = [c for c in article_page2.categories]

df_page1.to_csv('sabra_shatila.csv', index=False)
df_page2.to_csv('synthpop.csv', index=False)
# https://en.wikipedia.org/wiki/Sabra_and_Shatila_massacre