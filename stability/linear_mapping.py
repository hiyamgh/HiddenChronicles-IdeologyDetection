import os, pickle
from gensim.models import Word2Vec
from bias.utilities import get_edits_missing
from smart_open import open
from gensim.models import translation_matrix
from sklearn.decomposition import PCA
import plotly.graph_objects as go

# nltk for getting stopwords
from nltk.corpus import stopwords
stopwords_list = stopwords.words('arabic')
from utilities import cossim, file_to_list, check_terms


def stability_linear(w1, w2, transmat_12, transmat_21):
    """
    Args
        w1: word w in embedding space 1
        w2: word w in embedding space 2
        transmat_12: translation matrix to translate vector from embedding space 1 to embedding space 2
        transmat_21: translation matrix to translate vector from embedding space 2 to embedding space 1
    """
    def sim_12():
        """ cosine similarity of vector w to its mapped vector after mapping 2->1 then 1->2 """
        return cossim(transmat_21.dot(transmat_12).dot(w1), w1)

    def sim_21():
        """ cosine similarity of vector w to its mapped vector after mapping 1->2 then 2->1 """
        return cossim(transmat_12.dot(transmat_21).dot(w2), w2)

    return (sim_12() + sim_21()) / 2


# train_file = "OPUS_en_it_europarl_train_5K.txt"
#
# with open(train_file, "r") as f:
#     word_pair = [tuple(utils.to_unicode(line).strip().split()) for line in f]
# print (word_pair[:10])

# word pairs let them be arabic stop words

#
word2vecs_dir_nahar = 'E:/newspapers/word2vec/nahar/embeddings/'
word2vecs_dir_hayat = 'E:/newspapers/word2vec/hayat/embeddings/'
source_word2vec = Word2Vec.load(os.path.join(word2vecs_dir_nahar, 'word2vec_1967'))
target_word2vec = Word2Vec.load(os.path.join(word2vecs_dir_hayat, 'word2vec_1967'))
count = 0
print('number of arabic stopwords: {}'.format(len(stopwords_list)))

if not os.path.isfile('common_stopwords.p'):
    common_stopwords = []
    for stp in stopwords_list:
        if stp in source_word2vec.wv and stp in target_word2vec.wv:
            common_stopwords.append(stp)
            continue
        else:
            possibilities_source = get_edits_missing(stp, source_word2vec.wv)
            possibilities_target = get_edits_missing(stp, target_word2vec.wv)
            if possibilities_source != -1 and possibilities_target != -1:
                terms_in_both = list(set(possibilities_source).intersection(set(possibilities_target)))
                if terms_in_both:
                    print('{}: terms intersection in both: {}'.format(stp, terms_in_both))
                    for t in terms_in_both:
                        common_stopwords.append(t)
                    continue
                else:
                    print('{}: {} no intersection'.format(count, stp))
                    count += 1
            else:
                print('{}: {} not in both'.format(count, stp))
                count += 1

    print('stopwords coverage = {:.2f}%'.format(100 - (count/len(stopwords_list)) * 100))
    with open('common_stopwords.p', 'wb') as f:
        pickle.dump(common_stopwords, f)
else:
    with open('common_stopwords.p', 'rb') as f:
        common_stopwords = pickle.load(f)

word_pais = [(tok, tok) for tok in common_stopwords]
transmat12 = translation_matrix.TranslationMatrix(source_word2vec.wv, target_word2vec.wv, word_pais)
transmat12.train(word_pais)
print ("the shape of translation matrix 1>2 is: ", transmat12.translation_matrix.shape)

transmat21 = translation_matrix.TranslationMatrix(target_word2vec.wv, source_word2vec.wv, word_pais)
transmat21.train(word_pais)
print ("the shape of translation matrix 2>1 is: ", transmat21.translation_matrix.shape)

israel_list = file_to_list(txt_file='bias/israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt')
content_nahar = 'E:/newspapers/word2vec/nahar/embeddings/'
# content_hayat = 'E:/newspapers/word2vec/hayat/embeddings/'
model_nahar33 = Word2Vec.load(os.path.join(content_nahar, 'word2vec_{}'.format(1933)))
model_nahar67 = Word2Vec.load(os.path.join(content_nahar, 'word2vec_{}'.format(1967)))
found_nahar33, _ = check_terms(israel_list, model_nahar33)
found_nahar67, _ = check_terms(israel_list, model_nahar67)
common_words = list(set(found_nahar33) & set(found_nahar67))
for w in common_words:
    veca = model_nahar33.wv[w]
    vecb = model_nahar67.wv[w]
    print('{}: {}'.format(w, stability_linear(veca, vecb, transmat12.translation_matrix, transmat21.translation_matrix)))

# # The pair is in the form of (English, Italian), we can see whether the translated word is correct
# words = word_pais[:10]
# source_word, target_word = zip(*words)
# translated_word = transmat12.translate(source_word, 5, )
#
# for k, v in translated_word.items():
#     print ("word ", k, " and translated word", v)
#
# en_words_vec = [source_word2vec.wv[item[0]] for item in words]
# it_words_vec = [target_word2vec.wv[item[1]] for item in words]
#
# en_words, it_words = zip(*words)
#
# pca = PCA(n_components=2)
# new_en_words_vec = pca.fit_transform(en_words_vec)
# new_it_words_vec = pca.fit_transform(it_words_vec)
#
# # you can also using plotly lib to plot in one figure
# trace1 = go.Scatter(
#     x = new_en_words_vec[:, 0],
#     y = new_en_words_vec[:, 1],
#     mode = 'markers+text',
#     text = en_words,
#     textposition = 'top left'
# )
# trace2 = go.Scatter(
#     x = new_it_words_vec[:, 0],
#     y = new_it_words_vec[:, 1],
#     mode = 'markers+text',
#     text = it_words,
#     textposition = 'top left'
# )
# layout = go.Layout(
#     showlegend = False
# )
# data = [trace1, trace2]
#
# fig = go.Figure(data=data, layout=layout)
# fig.write_image("linear_mapping_example.png")



