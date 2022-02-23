import fasttext
import os


def mkdir(save_dir):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


def read_keywords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        words = f.readlines()
    words = [w[:-1] for w in words if '\n' in w]
    words = [w for w in words if w.strip() != '']
    return words


def generate_summaries(words, models_path, years, k, save_dir, file_name):
    for year in years:
        model1 = fasttext.load_model(os.path.join(models_path, '{}.bin'.format(year - 1)))  # embedding space 1
        model2 = fasttext.load_model(os.path.join(models_path, '{}.bin'.format(year)))  # embedding space 2

        years_included = {}

        for w in words:
            if w not in years_included:
                years_included[w] = []

            neighbors1 = set([out[1] for out in model1.get_nearest_neighbors(w, k)][:k])  # the 'summaries' for word w in embedding space 1
            neighbors2 = set([out[1] for out in model2.get_nearest_neighbors(w, k)][:k])  # the 'summaries' for word w in embedding space 2

            with open(os.path.join(save_dir, file_name + '.txt'), 'a', encoding='utf-8') as f:
                y1 = year - 1
                if y1 not in years_included[w]:
                    print('getting neighbors of {} in year {}'.format(w, y1))
                    f.write('\nplease manually correct the summaries below for the word: {} in year: {}\n'.format(w, year - 1))
                    summary = [n for n in neighbors1]  # words are the neighbors
                    for s in summary:
                        f.write('{}:\n'.format(s))
                    f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    years_included[w].append(y1)

                y2 = year
                if y2 not in years_included[w]:
                    print('getting neighbors of {} in year {}'.format(w, y2))
                    f.write(
                        '\nplease manually correct the summaries below for the word: {} in year: {}\n'.format(w, year))
                    summary = [n for n in neighbors2]  # words are the neighbors
                    for s in summary:
                        f.write('{}:\n'.format(s))
                    f.write('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    years_included[w].append(y2)


if __name__ == '__main__':
    years = list(range(1983, 2012))
    keywords = read_keywords('input/all_keywords2.txt')
    path2models = '/scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/'
    sumamries_folder = './'
    generate_summaries(words=keywords, models_path=path2models, years=years, k=20, save_dir=sumamries_folder, file_name='summaries_gonen2')
