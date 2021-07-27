import getpass


def get_text_dirs():
    if getpass.getuser() == '96171':
        TEXT_DIR1_nahar = 'E:/newspapers/nahar/nahar/nahar-batch-1/out/'
        TEXT_DIR2_nahar = 'E:/newspapers/nahar/nahar/nahar-batch-2/out/'
        TEXT_DIR3_nahar = 'E:/newspapers/nahar/nahar/nahar-batch-3/out/'
        TEXT_DIR4_nahar = 'E:/newspapers/nahar/nahar/nahar-batch-4/out/'

        # assafir archive
        TEXT_DIR1_assafir = 'E:/newspapers/assafir/assafir/assafir-batch-1/out/'
        TEXT_DIR2_assafir = 'E:/newspapers/assafir/assafir/assafir-batch-2/out/'

        # hayat archive
        TEXT_DIR1_hayat = 'E:/newspapers/hayat/hayat/hayat-batch-1/out/'
        TEXT_DIR2_hayat = 'E:/newspapers/hayat/hayat/hayat-batch-2/out/'

    else:
        # nahar archive
        TEXT_DIR1_nahar = 'G:/newspapers/nahar/nahar/nahar-batch-1/out/'
        TEXT_DIR2_nahar = 'G:/newspapers/nahar/nahar/nahar-batch-2/out/'
        TEXT_DIR3_nahar = 'G:/newspapers/nahar/nahar/nahar-batch-3/out/'
        TEXT_DIR4_nahar = 'G:/newspapers/nahar/nahar/nahar-batch-4/out/'

        # assafir archive
        TEXT_DIR1_assafir = 'G:/newspapers/assafir/assafir/assafir-batch-1/out/'
        TEXT_DIR2_assafir = 'G:/newspapers/assafir/assafir/assafir-batch-2/out/'

        # hayat archive
        TEXT_DIR1_hayat = 'G:/newspapers/hayat/hayat/hayat-batch-1/out/'
        TEXT_DIR2_hayat = 'G:/newspapers/hayat/hayat/hayat-batch-2/out/'

        # txt files directory for annahar
    TEXT_DIRS_nahar = [TEXT_DIR1_nahar, TEXT_DIR2_nahar, TEXT_DIR3_nahar, TEXT_DIR4_nahar]
    TEXT_DIRS_assafir = [TEXT_DIR1_assafir, TEXT_DIR2_assafir]
    TEXT_DIRS_hayat = [TEXT_DIR1_hayat, TEXT_DIR2_hayat]

    newspapers_dict = {
        'nahar': TEXT_DIRS_nahar,
        'assafir': TEXT_DIRS_assafir,
        'hayat': TEXT_DIRS_hayat,
    }

    return newspapers_dict