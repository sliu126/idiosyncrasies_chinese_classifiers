# On the Idiosyncrasies of the Mandarin Chinese Classifier System
Code and data for [On the Idiosyncrasies of the Mandarin Chinese Classifier System (NAACL 2019)](https://arxiv.org/abs/1902.10193).

## Code Dependencies
The code needs to be run in Python3. Packages that need to be installed are pickle, tqdm and nltk.

## Data Format
For the noun_classifier and noun_adj_classifier files, the data are pickled as Python Counter objects with keys being (noun, classifier) pairs and (noun, adjective, classifier) tuples, and values being the number of times each key appears in [our corpus](https://catalog.ldc.upenn.edu/LDC2005T14). The data is in traditional Mandarin Chinese. The supersense files are pickled as Python dictionaries with keys being noun/adjective supersenses and values being counters (in which counter keys being (noun, adjective, classifier) tuples under that specific sense and counter values being the number of times the tuples appear in our corpus).

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
