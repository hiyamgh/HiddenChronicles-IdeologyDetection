import pickle
import csv
import pandas as pd
import os


def mkdir(direc):
    if not os.path.exists(direc):
        os.makedirs(direc)

# Write CSV file
# with open("test.csv", "wt") as fp:
#     writer = csv.writer(fp, delimiter=",")
#     # writer.writerow(["your", "header", "foo"])  # write header
#     writer.writerows(data)
#

mapar2en = {
        'الولايات المتحدة الامريكية': 'UnitedStatesofAmerica',
        'امريكا': 'America',
        'اسرائيل': 'Israel',
        'فلسطيني': 'Palestinian',
        'حزب الله': 'Hezbollah',
        'المقاومه': 'Resistance',
        'سوري': 'Syrian',
        'منظمه التحرير الفلسطينيه': 'PalestinianLiberationOrganization',
        'ايران': 'Iran',
        'السعوديه': 'Saudiya'
    }

# words = ['اسرائيل', 'حزب الله']
# words = ['المقاومه', 'سوري']
words = list(mapar2en.keys())
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
modes = ['d-nahar', 'd-assafir', 'd-hayat']
for mode in modes:
    path = 'summaries_threshold/{}/'.format(mode)
    mkdir(path)
    for w in words:
        with open(os.path.join(path, "summaries_threshold_{}.csv".format(mapar2en[w])), "w", encoding="utf-8-sig", newline='') as f:
            writer = csv.writer(f, delimiter=",")
            w_all_summaries = {}
            for t in thresholds:
                w_all_summaries[t] = {}
                vps = []
                vwps = []
                with open('evaluate_stability/{}/all_summaries_{}.pickle'.format(mode, str(t)), 'rb') as handle:
                    all_summaries = pickle.load(handle)
                    for viewpoints in all_summaries[w]:
                        print('Viewpoint batch: {}'.format(viewpoints))
                        for vp in all_summaries[w][viewpoints]:
                            summary = all_summaries[w][viewpoints][vp]
                            if vp not in w_all_summaries[t]:
                                w_all_summaries[t][vp] = summary
                                vps.append(vp)
                            if viewpoints not in vwps:
                                vwps.append(viewpoints)

            with open('../simple_interpretable_usage_change/evaluate_stability_gonen/{}/summary_dict.pickle'.format(mode), 'rb') as handle:
                all_summaries_gonen = pickle.load(handle)

            print('finished getting all summaries for {} for each threshold'.format(w))
            for i in range(len(vps)):
                header = [str(t) for t in thresholds] + ['gonen'] + [vps[i]]
                writer.writerow(header)
                if i == len(vps) - 1:
                    for j in range(30):
                        list_init = [w_all_summaries[t][vps[i]][j] if j < len(w_all_summaries[t][vps[i]]) else '' for t in thresholds]
                        list_gonen = [all_summaries_gonen[w][vwps[i-1]][vps[i]][j]]
                        writer.writerow(list_init + list_gonen)
                    writer.writerow([''] * len(thresholds))
                    writer.writerow([''] * len(thresholds))
                else:
                    for j in range(30):
                        list_init = [w_all_summaries[t][vps[i]][j] if j < len(w_all_summaries[t][vps[i]]) else '' for t in thresholds]
                        list_gonen = [all_summaries_gonen[w][vwps[i]][vps[i]][j]]
                        writer.writerow(list_init + list_gonen)

                    writer.writerow([''] * len(thresholds))
                    writer.writerow([''] * len(thresholds))

        f.close()

        # convert the file into xlsx
        df = pd.read_csv(os.path.join(path, 'summaries_threshold_{}.csv'.format(mapar2en[w])))
        df.to_excel(os.path.join(path, 'summaries_threshold_{}.xlsx'.format(mapar2en[w])), index=False)


        # print('word = {} ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(w))
        # for t in thresholds:
        #     print('threshold = {} --------------------------------------'.format(t))
        #     with open('evaluate_stability/d-nahar/all_summaries_{}.pickle'.format(str(t)), 'rb') as handle:
        #         all_summaries = pickle.load(handle)
        #         for viewpoints in all_summaries[w]:
        #             print('Viewpoint batch: {}'.format(viewpoints))
        #             for vp in all_summaries[w][viewpoints]:
        #                 print('summary of {} from {} viewpoint'.format(w, vp))
        #                 for s in all_summaries[w][viewpoints][vp]:
        #                     print(s)


