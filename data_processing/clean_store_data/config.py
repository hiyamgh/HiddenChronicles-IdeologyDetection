'''
This code was taken and modified from the following repository: https://github.com/disooqi/ArabicProcessingCog
'''


# Sentence breaking
sentence_breaker = 0

# Stopwords

# Tokenization

# Normalization
normalizer = 0

ignore_oneword_line = True

Add_SPACE_after_TEH_MARBUTA = True

remove_diacritics = True

replace_TEH_MARBUTA = True
replace_ALEF_MAKSURA = True
remove_repeating_characters = False # here, the word ta2assasa for example تأسس , it has a repeating character
                                    # but its part of the original word

remove_nonArabic_word = False
process_nonArabic_word = True

replace_ALEF_HAMZA_ABOVE = True
replace_ALEF_HAMZA_BELOW = True
replace_ALEF_MADDA = True

remove_KASHIDA = True
remove_stopwords = True

'''
Punctuation:
in case remove_punc is True, isolate_punc and replace_punc have no effect
'''
remove_punc = True
isolate_punc = False
replace_punc = False

# Segmentation
segmenter = 0
# Stemming
stemmer = 0
# Lemmatization
lemmatizer = 0




