## About This Directory

Although I have trained fasttext on all models, but the following were giving the error:

``
  File "words_are_malleable4.py", line 735, in <module>
    model2 = fasttext.load_model(os.path.join(args.path2, args.model2))
  File "/home/hkg02/.local/lib/python3.7/site-packages/fasttext/FastText.py", line 436, in load_model
    return _FastText(model_path=path)
  File "/home/hkg02/.local/lib/python3.7/site-packages/fasttext/FastText.py", line 94, in __init__
    self.f.loadModel(model_path)
ValueError: /scratch/7613491_hkg02/political_discourse_mining_hiyam/Train_Word_Embedidng/fasttext/assafir/SGNS/ngrams4-size300-window5-mincount100-negative15-lr0.001/1989.bin has wrong file format!
``

I deleted old ones and re-trained these again:

- An-Nahar:
  - 1938.bin
  - 1959.bin
  - 1996.bin
- As-Safir:
  - 1989.bin
  - 1997.bin
  - 2001.bin
- Hayat
  - 1990.bin