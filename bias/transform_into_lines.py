import os


def print_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        all = f.readlines()
        s = ', '.join([s[:-1] for s in all])
        print(s)


if __name__ == '__main__':
    # participants_israel = 'participants_aspect/participants_israel.txt'
    # print_lines(participants_israel)

    # participants_palestine = 'participants_aspect/participants_palestine.txt'
    # print_lines(participants_palestine)

    # participants_other = 'participants_aspect/other_participants.txt'
    # print_lines(participants_other)

    # agents_violence = 'military_aspect/agents_of_violence.txt'
    # print_lines(agents_violence)

    # methods_violence = 'military_aspect/methods_of_violence.txt'
    # print_lines(methods_violence)

    # outcomes_violence = 'military_aspect/outcomes_of_violence.txt'
    # print_lines(outcomes_violence)

    # break_violence = 'military_aspect/break_from_violence.txt'
    # print_lines(break_violence)

    # occupation_practices = 'Israeli_Occupation_Practices/israeli_occupation_practices.txt'
    # print_lines(occupation_practices)

    occupation_practices = 'israeli_palestinian_conflict/occupations_vs_peace+israel/israel_list_arabic.txt'
    print_lines(occupation_practices)