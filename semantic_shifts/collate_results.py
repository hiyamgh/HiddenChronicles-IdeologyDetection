from pyexcel.cookbook import merge_all_to_a_book
# import pyexcel.ext.xlsx # no longer required if you use pyexcel >= 0.2.2
import glob

for year in range(2000, 2005):
    merge_all_to_a_book(glob.glob("C:/Users/hkg02/Downloads/results/results/{}/*.csv".format(year)), "output_{}.xlsx".format(year))