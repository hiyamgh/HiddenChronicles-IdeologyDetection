'''

We have stored all files in hdf5 format. This helped us store in each dataset in the file:
* The raw text
* The cleaned text after applying normalization
* meta data about the txt file stored (which is a page of a newspaper issue):
    - year
    - month
    - day
    - page number
This was done for each archive. For groupings we grouped by the year number. This will help
for navigation, and will take less time than looping over files in a directory in order to find
a certain issue in a certain year/day/ etc. Example of structure

____1995:
|_________95081102-raw:   ... ذهب الولد الى المدرسسسه
|_________95081102-clean:  ... ذهب الولد الى المدرسه
          |__year
          |__month
          |__day
          |__pagenb

|_________95081109-raw
|_________95081109-clean:
          |__year
          |__month
          |__day
          |__pagenb
'''