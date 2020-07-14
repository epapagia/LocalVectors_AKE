# Keyword Extraction

This is a repository providing code and datasets used in "Keywords lie far from the mean of all words in local vector space".

Updates will be released soon. Stay tuned.

## Quickstart

The classic stopwords list of the English language, as well as the lists of common adjectives, reporting verbs, determiners and functional words, are available in the corresponding .txt files:
stopwords_snowball_expanded.txt, common_adjectives.txt, reporting_verbs.txt, determiners.txt, functional_words.txt

The main code of our approach utilizes the [PKE](https://github.com/boudinfl/pke) package (i.e., LV_AKE.ipynb).

### Download datasets

```git clone https://github.com/boudinfl/ake-datasets.git```

Then, you should assign to the variable *dataset_path* the target dataset's path.

### Evaluation

We use the Functions.py to facilitate the evaluation process using the gold standard of authors' keywords. Precision/Recall/F1-measure are calculated in the LV_AKE.ipynb notebook.

### Citation

Please cite the following papers if you are interested in using our code.

```
@InProceedings{boudin:2016:COLINGDEMO,
  author    = {Boudin, Florian},
  title     = {pke: an open source python-based keyphrase extraction toolkit},
  booktitle = {Proceedings of COLING 2016, the 26th International Conference on Computational Linguistics: System Demonstrations},
  month     = {December},
  year      = {2016},
  address   = {Osaka, Japan},
  pages     = {69--73},
  url       = {http://aclweb.org/anthology/C16-2015}
}
```

```
@article{Papagiannopoulou2019outliers,
	author    = {Eirini Papagiannopoulou and Grigorios Tsoumakas},
	title     = {Unsupervised Keyphrase Extraction from Scientific Publications},
	journal = {To appear in Proceedings of the 20th International Conference on Computational Linguistics and Intelligent Text Processing, CICLing 2019},
	volume    = {La Rochelle, France},
    year      = {2019},
	url       = {https://arxiv.org/abs/1808.03712}
}
```