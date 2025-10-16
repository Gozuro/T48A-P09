from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics

# Cargar datos
categories = ['alt.atheism', 'comp.graphics', 'sci.space']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# Crear pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Entrenar
model.fit(train.data, train.target)

# Predecir
predicted = model.predict(test.data)

# Evaluar
print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
