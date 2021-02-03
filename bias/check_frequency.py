import h5py
import os
import getpass
import pickle
import matplotlib.pyplot as plt

''' check the frequency of mentions of the israeli-palestinian conflict
    per newspaper -- the frequency has a say on how much the media is highlighting the problem
'''

alphabet = list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B))
diactitics = list(range(0x064B, 0x0653))

alphabet = [chr(x) for x in alphabet]
diactitics = [chr(x) for x in diactitics]


def edits1(word):
    "All edits that are one edit away from `word`."
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in alphabet]
    inserts    = [L + c + R               for L, R in splits for c in alphabet]
    return list(set(deletes + transposes + replaces + inserts))


def edits2(word):
    ''' All edits that are two edit distances away from `word` '''
    return [e2 for e1 in edits1(word) for e2 in edits1(e1)]


def analyze_mentions_archives(mentions_dict):
    years_nahar = list(mentions_dict['nahar'].keys())
    years_hayat = list(mentions_dict['hayat'].keys())
    years_assafir = list(mentions_dict['assafir'].keys())

    years_common = list(sorted(list(set(years_nahar) & set(years_hayat) & set(years_assafir))))

    for archive in mentions_dict:
        y_mentions = [mentions_dict[archive][year] for year in years_common]
        plt.plot(years_common, y_mentions, marker='o', label=archive)

    plt.legend()
    plt.xlabel('Common Years')
    plt.ylabel('Number of mentions of Israeli-Palestinian Conflict')
    plt.show()


def create_mentions_dict():
    archives2count = {}
    w1 = 'فلسطين'
    w2 = 'فلسطينية'
    w3 = 'اسرائيل'
    w4 = 'اسرايلية'
    w1_edits1 = edits1(w1)
    w2_edits1 = edits1(w2)
    w3_edits1 = edits1(w3)
    w4_edits1 = edits1(w4)
    all_possibilities = w1_edits1 + w2_edits1 + w3_edits1 + w4_edits1

    archives = ['nahar', 'hayat', 'assafir']
    for archive in archives:
        print('archive: {}'.format(archive))
        if getpass.getuser() == '96171':
            h5_location = 'E:/newspapers/'
        else:
            h5_location = '../'

        if archive not in ['nahar', 'hayat', 'assafir']:
            raise ValueError('The requested archive is not found.'
                             ' You should choose one of the following: {}'.format(
                ['nahar', 'hayat', 'assafir']))

        hf = h5py.File('{}.h5'.format(os.path.join(h5_location, archive)), 'r')

        archives2count[archive] = {}

        for year in hf.keys():
            count = 0
            print('year: {}'.format(year))
            for issue in hf[year].keys():
                # if issue[-2:] == '01' or issue[-2:] == '02' or issue[-2:] == '03':
                content = hf[year][issue].value
                if any(all_possibilities) in content.split():
                    print('yes')
                    count += 1


            archives2count[archive][year] = count

    with open('mentions.p', 'wb') as handle:
        pickle.dump(archives2count, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':

    create_mentions_dict()

    with open('mentions.p', 'rb') as handle:
        mentions = pickle.load(handle)

    analyze_mentions_archives(mentions_dict=mentions)



