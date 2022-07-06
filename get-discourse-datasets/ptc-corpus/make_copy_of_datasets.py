import shutil
import os

if __name__ == '__main__':

    save_dir = 'annotated_by_discourse/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    labels_political_discourse = ['active', 'euphimism', 'details', 'exaggeration',	'bragging',	'litote', 'repetition',
                                  'metaphor', 'he said', 'apparent denial', 'apparent concession', 'blame transfer',
                                  'other kinds', 'opinion', 'irony']

    for file in os.listdir('.'):
        if file.endswith('.csv'):
            shutil.copy(file, save_dir)
            print('copied the file {} to {}'.format(file, save_dir))
