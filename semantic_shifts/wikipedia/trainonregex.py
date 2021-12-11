import re
def remove_file_instances(s):
    if '[[' in s and ']]' in s and 'File' in s:
        ind2 = s.rfind(']')
        return s[ind2+1:]
    elif '[[' in s and ']]' in s and 'Image' in s:
        ind2 = s.rfind(']')
        return s[ind2+1:]
    else:
        return s

def to_delete(s):
    # return 'name' in s or '{' in s or '}' in s or '=' in s or 'class' in s or 'nowrap' in s or s in ['title', 'image', 'flag', 'Flagicon', 'flagicon', 'small'] or '.png' in s or '.svg' in s or 'File' in s or 'px' in s or 'Image' in s or '<small>' in s or 'Historical' in s or 'date' in s or len(s) == 1 or '\"' in s or '(' in s or 'not in' in s or 'Nowrap' in s
    return 'name' in s or '=' in s or 'class' in s or 'nowrap' in s or s in ['title', 'image', 'flag', 'Flagicon', 'flagicon', 'small'] or '.png' in s or '.svg' in s or 'File' in s or 'px' in s or 'Image' in s or '<small>' in s or 'Historical' in s or 'date' in s or len(s) == 1 or '\"' in s or '(' in s or 'not in' in s or 'Nowrap' in s


def get_list_items(s):
    # s = re.sub('{{.*?}}', '', s)
    # print(1, s)
    s = re.sub('\'', '', s)
    print(2, s)
    s = re.sub(r'\[\[(?:[^\]|]*\|)?([^\]|]*)\]\]', r'\1', s)
    print(3, s)
    s = re.sub('\n----', '<br>', s)
    print(4, s)
    s = re.sub('\n\*', '<br>', s)
    print(5, s)
    s = re.sub('<br />', '<br>', s)
    print(6, s)
    s = re.sub('<br/>', '<br>', s)
    print(7, s)
    s = re.sub('</ref>|ref||</ref>', '', s)
    print(8, s)
    s = re.sub('</ref>', '', s)
    print(9, s)
    s = re.sub('ref', '', s)
    print(10, s)
#     s = re.sub('|', '', s)
    s = s.translate({ord('|'): '<br>'})
    print(11, s)
    s = s.translate({ord('{'): None, ord('}'): None})
    print(11, s)
#     if '<br>' in s:
    s = s.split("<br>")
    print(12, s)
#     else:
#         s = s.split("\|")
    s = [e.strip() for e in s if e.strip() != '']
    print(13, s)
    s = [remove_file_instances(e).strip() for e in s]
    print(14, s)
    s = [e for e in s if not to_delete(e)]
    print(15,  s)
    s = list(set(s)) # keep unique entries
    print('s is now: ', s)
    new_s = []
    for e in s:
        if ',' in e:
            splitted_e = e.split(',')
            for ee in splitted_e:
                new_s.append(ee)
        else:
            new_s.append(e)
    print('s is now after: ', new_s)
    return new_s

mystr = "{{Nowrap|[[Palestinian nationalism]]<br>[[Ba'athism#Neo-Ba'athism|Neo-Ba'athism]]<br>[[Ba'athism#Saddamism|Saddamism]]<br>[[Ba'athism]]}}"
get_list_items(mystr)