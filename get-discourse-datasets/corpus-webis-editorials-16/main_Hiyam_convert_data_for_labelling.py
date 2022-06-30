import os
import pandas as pd

editorial_portals = ['aljazeera', 'foxnews', 'guardian']
labels_political_discourse = ['active', 'euphimism', 'details', 'exaggeration',	'bragging',	'litote', 'repetition',
                              'metaphor', 'he said', 'apparent denial', 'apparent concession', 'blame transfer',
                              'other kinds', 'opinion', 'irony']
# df = pd.DataFrame(columns=['Sentence', 'Label', 'Portal'])
for portal in editorial_portals:
    sentences, labels = [], []
    path = 'annotated-txt/split-by-portal-final/{}/'.format(portal)
    df = pd.DataFrame(columns=['Sentence', 'Label'])
    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            print('processing file: {}'.format(file))
            lines = f.readlines()[1:] # always skip the first line because it contains only the title of the editorial (article)
            # for i, line in enumerate(lines[1:]):
            i = 0
            while i < len(lines):
                line = lines[i]
                if len(line.split('\t')) == 3:
                    num, label, sentence = line.split('\t')
                else:
                    # num, label, sentence, _ = line.split('\t') # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                    data = line.split('\t')
                    num = data[0]
                    label = data[1]
                    sentence = data[2]

                sentence = sentence.replace("\n", "")
                if label not in ["continued", "no-unit", "par-sep"]:
                    label_to_be = label
                    start_line_num = i
                    end_line_num = i
                    temp_sentences = []
                    temp_sentences.append(sentence)
                    for line in lines[i+1:]:
                        if len(line.split('\t')) == 3:
                            num, label, sentence = line.split('\t')
                        else:
                            # num, label, sentence, _ = line.split('\t')  # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                            data = line.split('\t')
                            num = data[0]
                            label = data[1]
                            sentence = data[2]

                        sentence = sentence.replace("\n", "")
                        # if label == "continued" or (label == "no-unit" and sentence != "."):
                        if not (label == "no-unit" and sentence == "."):
                            temp_sentences.append(sentence)
                            end_line_num += 1
                            continue
                        else: # this is the no-unit with "."
                            temp_sentences.append(sentence)
                            end_line_num += 1
                            whole_sentence = " ".join(temp_sentences)

                            sentences.append(whole_sentence)
                            labels.append(label_to_be)
                            # portals.append(portal)

                            i = end_line_num + 1
                            break
                    # will reach this point if no more lines
                    # temp_sentences.append(sentence)
                    # end_line_num += 1
                    if end_line_num + 1 >= len(lines):
                        whole_sentence = " ".join(temp_sentences)

                        sentences.append(whole_sentence)
                        labels.append(label_to_be)
                        # portals.append(portal)
                        i = end_line_num + 1

                elif label == "continued":
                    start_line_num = i
                    end_line_num = i
                    temp_sentences = []
                    temp_sentences.append(sentence)
                    for line in lines[i + 1:]:
                        if len(line.split('\t')) == 3:
                            num, label, sentence = line.split('\t')
                        else:
                            # num, label, sentence, _ = line.split('\t')  # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                            data = line.split('\t')
                            num = data[0]
                            label = data[1]
                            sentence = data[2]

                        sentence = sentence.replace("\n", "")
                        # if label == "continued" or (label == "no-unit" and sentence != "."):
                        if not (label == "no-unit" and sentence == "."):
                            temp_sentences.append(sentence)
                            label_to_be = label
                            end_line_num += 1
                            continue
                        else:  # this is the no-unit with "." or par-sep
                            temp_sentences.append(sentence)
                            end_line_num += 1
                            whole_sentence = " ".join(temp_sentences)

                            sentences.append(whole_sentence)
                            labels.append(label_to_be)
                            # portals.append(portal)

                            i = end_line_num + 1
                            break
                    # will reach this point if no more lines
                    # temp_sentences.append(sentence)
                    # end_line_num += 1
                    if end_line_num + 1 >= len(lines):
                        whole_sentence = " ".join(temp_sentences)

                        sentences.append(whole_sentence)
                        labels.append(label_to_be)
                        # portals.append(portal)
                        i = end_line_num + 1
                elif label == "no-unit": # label is no-unit
                    tart_line_num = i
                    end_line_num = i
                    temp_sentences = []
                    labels_to_be = []
                    temp_sentences.append(sentence)
                    for line in lines[i + 1:]:
                        if len(line.split('\t')) == 3:
                            num, label, sentence = line.split('\t')
                        else:
                            # num, label, sentence, _ = line.split('\t')  # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                            data = line.split('\t')
                            num = data[0]
                            label = data[1]
                            sentence = data[2]

                        # sentence = sentence[:-1] if sentence[-1] == '\n' else sentence
                        sentence = sentence.replace('\n', '')
                        # if label == "continued" or (label == "no-unit" and sentence != "."):
                        if not (label == "no-unit" and sentence == "."):
                            temp_sentences.append(sentence)
                            labels_to_be.append(label)
                            end_line_num += 1
                            continue
                        else:  # this is the no-unit with "." or par-sep
                            temp_sentences.append(sentence)
                            end_line_num += 1
                            whole_sentence = " ".join(temp_sentences)

                            labels_to_be = list(set(labels_to_be))
                            final_label = "no-unit"
                            for label in labels_to_be:
                                if label != final_label and label != "continued":
                                    final_label = label
                                    break
                            sentences.append(whole_sentence)
                            labels.append(final_label)
                            # portals.append(portal)

                            i = end_line_num + 1
                            break
                    # will reach this point if no more lines
                    # temp_sentences.append(sentence)
                    # end_line_num += 1
                    if end_line_num + 1 >= len(lines):
                        whole_sentence = " ".join(temp_sentences)

                        sentences.append(whole_sentence)
                        labels.append(label_to_be)
                        # portals.append(portal)
                        i = end_line_num + 1
                else:
                    i += 1 # label is par-sep
                    continue

                print('i = {}'.format(i))
                if i == 215:
                    print('hello')
        print('finished processing {}'.format(file))

    df['Sentence'] = sentences
    df['Label'] = labels
    # df['Portal'] = portals
    for label in labels_political_discourse:
        df[label] = ""

    # df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('ArgumentationDataset_politicaldiscourse_{}.csv'.format(portal), index=False)
    df.to_excel('ArgumentationDataset_politicaldiscourse_{}.xlsx'.format(portal), index=False)

                # if label == 'par-sep':
                #     i += 1
                #     continue
                #
                # if label == 'no-unit':
                #     i += 1
                #     continue
                #
                # sentence = sentence[:-1] if sentence[-1] == '\n' else sentence
                #
                # if label == 'continued': # or label == 'no-unit'
                #     start_line_num = i
                #     end_line_num = i
                #     temp_sentences = []
                #     temp_sentences.append(sentence)
                #     for line in lines[i+1:]:
                #         if len(line.split('\t')) == 3:
                #             num, label, sentence = line.split('\t')
                #         else:
                #             num, label, sentence, _ = line.split('\t')  # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                #
                #         sentence = sentence[:-1] if sentence[-1] == '\n' else sentence
                #
                #         if label == 'continued':
                #             temp_sentences.append(sentence)
                #             end_line_num += 1
                #             continue
                #         else:
                #             temp_sentences.append(sentence)
                #             end_line_num += 1
                #             whole_sentence = ' '.join(temp_sentences)
                #
                #             sentences.append(whole_sentence)
                #             labels.append(label)
                #             portals.append(portal)
                #
                #             i = end_line_num + 1
                #             break
                #
                # else: # label is not no-unit and is not continued ==> one of the tags, keep adding sentences until you hit a no-unit with a "."
                #     if label == 'no-unit' and sentence != ".":
                #         start_line_num = i
                #         end_line_num = i
                #         temp_sentences = []
                #         temp_sentences.append(sentence)
                #         label_to_be = label
                #         for line in lines[i+1:]:
                #             if len(line.split('\t')) == 3:
                #                 num, label, sentence = line.split('\t')
                #             else:
                #                 num, label, sentence, _ = line.split('\t')  # not sure what is MOD-NU, I guess a label when the sentence has a continuation
                #
                #             sentence = sentence[:-1] if sentence[-1] == '\n' else sentence
                #
                #             # if label == 'continued' or (label == 'no-unit' and sentence != "."):
                #             if label == 'continued' or label == 'no-unit':
                #                 temp_sentences.append(sentence)
                #                 end_line_num += 1
                #                 continue
                #
                #             else: # encountered a no-unit with "."
                #                 temp_sentences.append(sentence) # this is the "."
                #                 end_line_num += 1
                #                 whole_sentence = ' '.join(temp_sentences)
                #
                #                 sentences.append(whole_sentence)
                #                 labels.append(label_to_be)
                #                 portals.append(portal)
                #
                #                 i = end_line_num + 1
                #                 break
                #
                #         # sentences.append(sentence)
                #         # labels.append(label)
                #         # portals.append(portal)
                #         i += 1
    # print()
