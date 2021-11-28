import argparse
import os
# a list of tuples, first and second elements in a tuple are the start and end year (inclusive)
# of the timeline, the third tuple is the list of newspapers that cover this timeline
timelines = [
    (1948, 1956, ['nahar']),
    (1956, 1958, ['nahar', 'hayat']),
    (1958, 1967, ['nahar', 'hayat']),
    (1967, 1969, ['nahar', 'hayat']),
    (1969, 1973, ['nahar', 'hayat']),
    (1973, 1975, ['nahar', 'hayat']),
    (1975, 1978, ['nahar', 'assafir']),
    (1978, 1982, ['nahar', 'assafir']),
    (1982, 1984, ['nahar', 'assafir']),
    (1984, 1987, ['nahar', 'assafir']),
    (1987, 1993, ['nahar', 'assafir']),
    (1993, 1996, ['nahar', 'hayat', 'assafir']),
    (1996, 2000, ['nahar', 'hayat', 'assafir']),
    (2000, 2001, ['nahar', 'assafir']),
    (2001, 2003, ['nahar', 'assafir']),
    (2003, 2005, ['nahar', 'assafir']),
    (2005, 2006, ['nahar', 'assafir']),
    (2006, 2008, ['nahar', 'assafir']),
    (2008, 2010, ['assafir']),
    (2010, 2011, ['assafir'])
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_dir', type=str, default='data_timelined/', help="output directory to save data files in")
    args = parser.parse_args()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for t in timelines:
        start_year, end_year = t[0], t[1]
        newspapers = t[2]

        for t in newspapers:
            for archive in newspapers:
                path = os.path.join(args.out_dir, '{}/'.format(archive))
                if not os.path.exists(path):
                    os.makedirs(path)

                fout = open(os.path.join(path, "{}_{}.txt".format(start_year, end_year)), "w")

                num_lines = 0
                nums_lines_per_file = []
                for y in range(start_year, end_year + 1):
                    fin = open("data/{}/{}.txt".format(archive, y), "r")
                    fout.write(fin.read())
                    n = sum(1 for line in fin)
                    num_lines += n
                    nums_lines_per_file.append(n)
                fout.close()
                assert num_lines == sum(nums_lines_per_file)
