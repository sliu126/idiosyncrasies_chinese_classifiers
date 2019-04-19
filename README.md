# On the Idiosyncrasies of the Mandarin Chinese Classifier System
Code and data for [On the Idiosyncrasies of the Mandarin Chinese Classifier System (NAACL 2019)](https://arxiv.org/abs/1902.10193).

## Code Dependencies
The code needs to be run in Python3. Packages that need to be installed are pickle, tqdm and nltk.

## Running Experiments
To compute the mutual information between classifiers and nouns, run
```
python compute_mutual_information.py -ICN
```

To compute the mutual information between classifiers and adjectives, run
```
python compute_mutual_information.py -ICA
```

To compute the mutual information between classifiers and noun supersenses, run
```
python compute_mutual_information.py -ICNS
```

To compute the mutual information between classifiers and adjective supersenses, run
```
python compute_mutual_information.py -ICAS
```

To compute the mutual information between classifiers and noun synsets, run
```
python compute_mutual_information.py -ICS
```
