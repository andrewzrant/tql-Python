```
def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filtered_tokens = filter(None,[re.sub(pattern=pattern,repl="",string=token) for token in tokens])
    filtered_text = ''.join(filtered_tokens)
    return filtered_text
    
def normalize_corpus(corpus):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        text = remove_stopwords(text)
        normalized_corpus.append(text)
    return normalized_corpus
```

```
def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
```
```
def tfidf_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features
```

```
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', n_iter=100)
    lr = LogisticRegression()


```



# 翻译
https://www.cnblogs.com/fanyang1/p/9414088.html