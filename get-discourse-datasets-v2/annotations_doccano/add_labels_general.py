import pandas as pd
import os

directories = [
    'translationsv2/van_dijk_contexts/Palestinian Resistance South Lebanon_cleaned/',
    'translationsv2/van_dijk_contexts/Sabra and Shatila Massacre_cleaned/'
]

labels2gen = {
    'Main': 'main_contents',
    'Main_Consequence': 'main_contents',
    'Cause_Specific': 'context_informing_contents',
    'Cause_General': 'context_informing_contents',
    'Distant_Historical': 'additional_supportive_contents',
    'Distant_Anecdotal': 'additional_supportive_contents',
    'Distant_Evaluation': 'additional_supportive_contents',
    'Distant_Expectations_Consequences': 'additional_supportive_contents',
    'NA': 'NA'
}

for folder in directories:
    for file in os.listdir(folder):
        labels_general = []
        df = pd.read_excel(os.path.join(folder, file))
        labels_specific = list(df['label'])
        for lbl in labels_specific:
            labels_general.append(labels2gen[lbl])
        df['label_general'] = labels_general
        df.to_excel(os.path.join(folder, file), index=False)

