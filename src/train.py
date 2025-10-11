from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
#from model_functions import create_model, train_model, predict, evaluate_metrics  # si las tienes en otro archivo

# Cargar datos
categories = ['alt.atheism', 'comp.graphics', 'sci.space']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

model = create_model()
model = train_model(model, train.data, train.target)

# Crear pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

predicted = predict(model, test.data)

# Entrenar
model.fit(train.data, train.target)

results = evaluate_metrics(test.target, predicted, average='macro')
print("\n📊 Resultados del modelo:")
for metric, value in results.items():
    print(f"{metric.capitalize()}: {value:.4f}")
# Predecir
predicted = model.predict(test.data)
print("\n📋 Reporte de clasificación completo:")
# Evaluar
print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
