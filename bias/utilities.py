import numpy as np
from gensim.models import Word2Vec
from itertools import *
import re
import os


alphabet = list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B))
diactitics = list(range(0x064B, 0x0653))

alphabet = [chr(x) for x in alphabet]
diactitics = [chr(x) for x in diactitics]


def mkdir(directory):
    ''' create a directory if it does not already exist '''
    if not os.path.exists(directory):
        os.makedirs(directory)


def cossim(w1, w2):
    ''' calculates cosine similarity between 2 vectors (transform words into vectors before using this function) '''
    cos_sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
    return cos_sim


def calc_distance_between_vectors(vec1, vec2, distype= 'norm'):
    ''' calculates distance between two vectors  (transform words into vectors before using this function) '''
    if distype is 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)


def file_to_list(txt_file):
    ''' gets the list of words from txt file'''
    if not os.path.exists(txt_file):
        raise ValueError('File {} does not exist'.format(txt_file))
    with open(txt_file, 'r', encoding="utf-8") as f:
        return f.read().splitlines()


def get_mean_vector(word2vec_model, words):
    ''' gets the representative vector from a list of vectors '''
    # remove out-of-vocabulary words
    words = [w for w in words if w in word2vec_model.wv]
    out_of_vocabulary = [w for w in words if w not in word2vec_model.wv]
    if out_of_vocabulary:
        print('Removed the following words (Out Of Vocabulary): {}'.format(out_of_vocabulary))
    if len(words) > 1:
        return np.mean(word2vec_model[words], axis=0)
    else:
        if len(words) == 1:
            return word2vec_model[words[0]]
        else:
            raise ValueError('all words are out of vocabulary')


def get_decades(decades_path):
    pattern = r'word2vec_\d{4}_\d{4}'
    matches = re.findall(pattern, ','.join(os.listdir(decades_path)))
    matches = list(set(matches))
    decades = []
    for match in matches:
        decade_pair = match[-9:]
        pair1 = decade_pair.split('_')[0]
        pair2 = decade_pair.split('_')[1]
        decades.append((pair1, pair2))
    return decades


def load_models(start_year, end_year, archive_path):
    # word2vec_models = []
    word2vec_models_dict = {}
    for year in range(int(start_year), int(end_year) + 1):
        desired_model_path = '{}/word2vec_{}'.format(archive_path, year)
        if os.path.exists(desired_model_path):
            # word2vec_models.append(Word2Vec.load('{}/word2vec_{}'.format(archive_path, year)))
            word2vec_models_dict[year] = Word2Vec.load(desired_model_path)
        else:
            print('skipping loading {} as it does not exist in the archive_path specified')

    return word2vec_models_dict


def get_min_max_years(desired_archives, archive_path):
    years = []
    for archive in desired_archives:
        for file in os.listdir(archive_path[archive]):
            if not file.endswith('.npy'):
                years.append(int(file.split('_')[1]))

    return min(years), max(years)


def get_archive_year(archive, archive_path):
    years = []
    for file in os.listdir(archive_path[archive]):
        if not file.endswith('.npy'):
            years.append(int(file.split('_')[1]))
    return years


def load_model_by_year(archive_path, target_year):
    for file in os.listdir(archive_path):
        if not file.endswith('.npy'):
            desired_model_path = os.path.join(archive_path, file)
            year = int(file.split('_')[1])
            if year == target_year:
                return Word2Vec.load(desired_model_path)


def load_all_models(archive_path):
    word2vec_models_dict = {}
    for file in os.listdir(archive_path):
        if not file.endswith('.npy'):
            desired_model_path = os.path.join(archive_path, file)
            year = int(file.split('_')[1])
            word2vec_models_dict[year] = Word2Vec.load(desired_model_path)

    return word2vec_models_dict


# D:\word2vec_decades\nahar\meta_data
def load_model_decade_level(decades_path, s_year, e_year):
    desired_model_path = os.path.join(decades_path, 'word2vec_{}_{}'.format(s_year, e_year))
    if os.path.exists(desired_model_path):
        return Word2Vec.load(desired_model_path)
    else:
        raise ValueError('The requested decade {}_{} is not found in {}'.format(s_year, e_year, desired_model_path))


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


def get_edits_missing(t, vocab):
    possibilities = edits1(t)
    if any(possibilities) in vocab:
        print('got the words that are 1 edit distance away from {}'.format(t))
        possibilities = [p for p in possibilities if p in vocab]
        return possibilities
    else:
        possibilities = edits2(t)
        if any(possibilities) in vocab:
            print('got the words that are 2 edit distances away from {}'.format(t))
            return possibilities
        else:
            return -1


def get_possibilities(word_list, vocab):
    populated = []
    for t in word_list:
        possibles = get_edits_missing(t, vocab)
        if possibles != -1:
            populated.extend(possibles)
    return populated



def check_terms(target_terms, word2vec_currmodel):
    ''' check if the target terms all exist in the vocabulary -- if some don't, return modified list '''
    vocabulary = word2vec_currmodel.wv
    found = [t for t in target_terms if t in vocabulary]
    not_found = [t for t in target_terms if t not in vocabulary]

    return found, not_found


def get_terms_missing_difference_percentage(original_terms, filtered_terms):
    ''' get the percentage of missing from the original vs filtered terms '''
    return (1 - len(filtered_terms)/len(original_terms)) * 100


def apply_augmentation(larger_list, smaller_list, word2vec_currmodel, topK):
    ''' augment smaller list until its size is equal to the size of the larger list '''
    augmented_list = smaller_list
    if len(smaller_list) + topK >= len(larger_list):
        pass
    else:
        topK = len(larger_list) - len(smaller_list)

    for t in smaller_list:
        most_similar = word2vec_currmodel.wv.most_similar(positive=[t], topn=topK)
        print('¬¬¬¬¬¬ term: {}'.format(t))
        for neighbor, sim in most_similar:
            print('neighbor: {}: sim: {}'.format(neighbor, sim))
            # print('sim: {}'.format(sim))
            augmented_list.append(neighbor)
        if len(augmented_list) >= len(larger_list):
            break

    augmented_list = list(set(augmented_list))
    print('augmentation step done ...')

    if len(augmented_list) == len(larger_list):
        return augmented_list

    else:
        while len(larger_list) < len(augmented_list):
            print('popping up ...')
            augmented_list.pop(-1)
        return augmented_list
    # else:
    #     apply_augmentation(larger_list, augmented_list, word2vec_currmodel, topK)

    # elif len(augmented_list) > len(larger_list):
    #     while len(larger_list) < len(augmented_list):
    #         print('popping up ...')
    #         augmented_list.pop(-1)
    #     return augmented_list
    # else:
    #     apply_augmentation(larger_list, augmented_list, word2vec_currmodel, topK)


def populate_list(target_terms, model, vocab, topK):
    populated = []
    for t in target_terms:
        if t in vocab:
            populated.append(t)
            tedit1 = edits1(t)
            tedit1 = [te for te in tedit1 if te in vocab]

            tedit1sorted = {}
            for te in tedit1:
                tedit1sorted[te] = model.similarity(t, te)
            tedit1sorted = {k: v for k, v in sorted(tedit1sorted.items(), key=lambda item: item[1], reverse=True)}
            top_edits = dict(islice(tedit1sorted.items(), topK))
            # print('\n top neighbors for {}:'.format(t))
            # for k, v in top_edits.items():
            #     print('{}: {}'.format(k, v))
            populated = populated + list(top_edits.keys())
        else:
            possible_from_missing = get_edits_missing(t, vocab)
            if possible_from_missing != -1:
                # print('\n top neighbors for {} (did not find t, getting its edits that exist in vocab):'.format(t))
                # print('edits that exist: {}'.format(possible_from_missing))
                populated = populated + possible_from_missing
            else:
                # print('skipping {} because neither it nor its edits are in the vocab'.format(t))
                pass

    return populated
