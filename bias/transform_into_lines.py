
def print_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        all = f.readlines()
        s = ', '.join([s[:-1] for s in all])
        print(s)


if __name__ == '__main__':
    participants_israel = 'keywords/arabic/participants_Israel_arabic.txt'
    print_lines(participants_israel)
    print('==================================')
    participants_palestine = 'keywords/arabic/participants_palestine_arabic.txt'
    print_lines(participants_palestine)
    print('==================================')
    terrorism = 'keywords/arabic/terrorism(100yearsofbias)_arabic.txt'
    print_lines(terrorism)
    print('==================================')
    methods_violence = 'keywords/arabic/methods_of_violence_arabic.txt'
    print_lines(methods_violence)
    print('==================================')
    peace_practices = 'keywords/arabic/non_occupation_practices_arabic.txt'
    print_lines(peace_practices)
    print('==================================')