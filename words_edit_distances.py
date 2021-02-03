def edits1(word, arabic_alphabet):
    "All edits that are one edit away from `word`."
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in arabic_alphabet]
    inserts    = [L + c + R               for L, R in splits for c in arabic_alphabet]
    return set(deletes + transposes + replaces + inserts)

# print(edits1('hiyam'))

alphabet = list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B))
diactitics = list(range(0x064B, 0x0653))

alphabet = [chr(x) for x in alphabet]
diactitics = [chr(x) for x in diactitics]

print(edits1('فلسطين', arabic_alphabet=alphabet))
print(edits1('الفلسطينيه', arabic_alphabet=alphabet))