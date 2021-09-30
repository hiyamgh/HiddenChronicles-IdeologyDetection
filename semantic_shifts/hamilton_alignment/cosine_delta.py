import os
import time
import random
import argparse
from multiprocessing import Process, Queue
from gensim.models import Word2Vec
import ioutils
import alignment


def oov(w2v_model, word):
    ''' function to check if wpr is out of vocabulary '''
    return word not in w2v_model.wv.vocab


def get_cosine_deltas(base_embeds, delta_embeds, words):
    deltas = {}
    base_embeds, delta_embeds = alignment.intersection_align_gensim(base_embeds, delta_embeds)
    for word in words:
        # if base_embeds.oov(word) or delta_embeds.oov(word):
        if oov(base_embeds, word) or oov(delta_embeds, word):
            deltas[word] = float('nan')
        else:
            # get cosine distance
            delta = base_embeds.wv[word].dot(delta_embeds.wv[word].T)
            deltas[word] = delta
    return deltas


def merge(out_pref, years, word_list):
    vol_yearstats = {}
    disp_yearstats = {}
    for word in word_list:
        vol_yearstats[word] = {}
        disp_yearstats[word] = {}
    for year in years:
        vol_yearstat = ioutils.load_pickle(out_pref + str(year) + "-vols.pkl")
        disp_yearstat = ioutils.load_pickle(out_pref + str(year) + "-disps.pkl")
        for word in word_list:
            if word not in vol_yearstat:
                vol = float('nan')
            else:
                vol = vol_yearstat[word]
            if word not in disp_yearstat:
                disp = float('nan')
            else:
                disp = disp_yearstat[word]
            vol_yearstats[word][year] = vol
            disp_yearstats[word][year] = disp
        os.remove(out_pref + str(year) + "-vols.pkl")
        os.remove(out_pref + str(year) + "-disps.pkl")
    ioutils.write_pickle(vol_yearstats, out_pref + "vols.pkl")
    ioutils.write_pickle(disp_yearstats, out_pref + "disps.pkl")


def worker(proc_num, queue, out_pref, in_dir, target_lists, displacement_base, year_inc):
    time.sleep(10*random.random())
    while True:
        if queue.empty():
            print(proc_num, "Finished")
            break
        year = queue.get()
        print(proc_num, "Loading matrices...")
        # base = create_representation(type, in_dir + str(year-year_inc),  thresh=thresh, restricted_context=context_lists[year], normalize=True, add_context=False)
        # delta = create_representation(type, in_dir + str(year),  thresh=thresh, restricted_context=context_lists[year], normalize=True, add_context=False)
        base = Word2Vec.load(os.path.join(in_dir, 'word2vec_{}'.format(year - year_inc)))
        delta = Word2Vec.load(os.path.join(in_dir, 'word2vec_{}'.format(year)))
        print(proc_num, "Getting deltas...")
        # year_vols = get_cosine_deltas(base, delta, target_lists[year], type)
        # year_disp = get_cosine_deltas(displacement_base, delta, target_lists[year], type)
        year_vols = get_cosine_deltas(base, delta, target_lists)
        year_disp = get_cosine_deltas(displacement_base, delta, target_lists)
        print(proc_num, "Writing results...")
        ioutils.write_pickle(year_vols, out_pref + str(year) + "-vols.pkl")
        ioutils.write_pickle(year_disp, out_pref + str(year) + "-disps.pkl")


# def run_parallel(num_procs, out_pref, in_dir, years, target_lists, context_lists, displacement_base, thresh, year_inc, type):
def run_parallel(num_procs, out_pref, in_dir, years, target_lists, displacement_base, year_inc):
    queue = Queue()
    for year in years:
        queue.put(year)
    procs = [Process(target=worker, args=[i, queue, out_pref, in_dir, target_lists, displacement_base, year_inc]) for i in range(num_procs)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    print("Merging")
    full_word_set = set([])
    for year_words in target_lists.itervalues():
        full_word_set = full_word_set.union(set(year_words))
    merge(out_pref, years, list(full_word_set))


if __name__ == '__main__':
    start_year = 1933 # start year
    end_year = 2005 # end year
    year_inc = 1 # increments between one embedding and another
    disp_year = 2005 # year to measure displacement from
    dir = 'F:/newspapers/word2vec/nahar/embeddings/' # in_dir
    out_dir = 'D:/results_deltas/'
    num_procs = 4 # number of processes to spawn
    type=''
    with open('words.txt', 'r', encoding='utf-8') as f:
        target_lists = f.readlines()
    # parser = argparse.ArgumentParser(description="Computes semantic change statistics for words.")
    # parser.add_argument("dir", help="path to word vectors")
    # parser.add_argument("num_procs", type=int, help="number of processes to spawn")
    # parser.add_argument("word_file", help="path to sorted word file")
    # parser.add_argument("out_dir", help="output path")
    # parser.add_argument("--target-words", type=int, help="Number of words (of decreasing average frequency) to analyze", default=-1)
    # parser.add_argument("--context-words", type=int, help="Number of words (of decreasing average frequency) to include in context. -2 means all regardless of word list", default=-1)
    # parser.add_argument("--context-word-file")
    # parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1800)
    # parser.add_argument("--year-inc", type=int, help="year increment", default=10)
    # parser.add_argument("--type", default="PPMI")
    # parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=2000)
    # parser.add_argument("--disp-year", type=int, help="year to measure displacement from", default=2000)
    # args = parser.parse_args()
    # years = range(args.start_year, args.end_year + 1, args.year_inc)

    years = range(start_year, end_year + 1, year_inc)
    # target_lists, context_lists = ioutils.load_target_context_words(years, args.word_file, args.target_words, -1)
    # if args.context_word_file != None:
    #     print "Loading context words.."
    #     _ , context_lists = ioutils.load_target_context_words(years, args.word_file, -1, args.context_words)
    # target_lists, context_lists = ioutils.load_target_context_words(years, args.word_file, args.target_words, args.context_words)

    ioutils.mkdir(out_dir)
    # displacement_base = create_representation(args.type, args.dir + "/" +  str(args.disp_year), restricted_context=context_lists[args.disp_year], normalize=True, add_context=False)

    displacement_base = Word2Vec.load(os.path.join(dir, 'word2vec_{}'.format(disp_year)))
    run_parallel(num_procs, out_dir, dir, years[1:], target_lists, displacement_base, year_inc)
