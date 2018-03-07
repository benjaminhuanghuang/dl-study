from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# define the category map
category_map = {
    'talk.politics.misc': 'Politics',
    'rec.autos': 'Autos',
    'rec.sport.hockey': 'Kockey',
    'sci.electronics' : 'Electronics',
    'sci.med' : 'Medicine'
}

# get the trainning dataset
trainning_data = fetch_20newsgroups(subset='train', categories=category_map, shuffle=True, random_state=5)


# buiild a count vectorizer and extract term counts
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(trainning_data)
print ('\nDimensions of training data:', train_tc.shape)

# create the tf-idf transformer
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)


