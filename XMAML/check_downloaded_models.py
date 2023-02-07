import glob
import json
import re
from collections import OrderedDict
from transformers import TRANSFORMERS_CACHE

metaFiles = glob.glob(TRANSFORMERS_CACHE + '/*.json')
modelRegex = "huggingface\.co\/(.*)(pytorch_model\.bin$|resolve\/main\/tf_model\.h5$)"

cachedModels = {}
cachedTokenizers = {}
for file in metaFiles:
    with open(file) as j:
        data = json.load(j)
        isM = re.search(modelRegex, data['url'])
        if isM:
            cachedModels[isM.group(1)[:-1]] = file
        else:
            cachedTokenizers[data['url'].partition('huggingface.co/')[2]] = file

cachedTokenizers = OrderedDict(sorted(cachedTokenizers.items(), key=lambda k: k[0]))

for k,v in cachedTokenizers.items():
    print(k, v)