from bias.utilities import file_to_list, check_terms
from gensim.models import Word2Vec
import getpass
from bias.utilities import get_min_max_years, get_archive_year
from bias.plotting import *

israel_list = file_to_list(txt_file='bias/israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt')

if getpass.getuser() == '96171':
    archives_wordembeddings = {
        'nahar': 'E:/newspapers/word2vec/nahar/embeddings/',
        'hayat': 'E:/newspapers/word2vec/hayat/embeddings/',
        'assafir': 'E:/newspapers/word2vec/assafir/embeddings/'
    }
else:
    archives_wordembeddings = {
        'nahar': 'D:/word2vec/nahar/embeddings/',
        'hayat': 'D:/word2vec/hayat/embeddings/',
        'assafir': 'D:/word2vec/assafir/embeddings/'
    }

archives_year = {
    'nahar': get_archive_year('nahar', archives_wordembeddings),
    'hayat': get_archive_year('hayat', archives_wordembeddings),
    'assafir': get_archive_year('assafir', archives_wordembeddings)
}

if __name__ == '__main__':
    archives_count = {}
    desired_archives = ['nahar', 'hayat', 'assafir']
    min_year, max_year = get_min_max_years(desired_archives, archives_wordembeddings)
    all_years = list(range(min_year, max_year + 1))
    for archive in desired_archives:
        archives_count[archive] = {}
        archives_count[archive]['years'] = all_years
        archives_count[archive]['counts'] = []
        for year in all_years:
            year_count = 0
            if year in archives_year[archive]:
                model = Word2Vec.load(os.path.join(archives_wordembeddings[archive], 'word2vec_{}'.format(year)))
                found, _ = check_terms(israel_list, model)
                if found:
                    for w in found:
                        year_count += model.wv.vocab[w].count
                else:
                    year_count = None
                archives_count[archive]['counts'].append(year_count)

    plot_counts(archives_count, 'latest', 'counts_israeli')