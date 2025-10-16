from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from src.model import create_model
from src.metrics import evaluate_metrics
from model import create_model
from metrics import evaluate_metrics


# Cargar datos
categories = ['alt.atheism', 'comp.graphics', 'sci.space']
train = fetch_20newsgroups(subset='train', categories=categories)
test = fetch_20newsgroups(subset='test', categories=categories)

# Crear pipeline
model = create_model()
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Entrenar
train_model(model, train.data, train.target)
model.fit(train.data, train.target)

# Predecir
predicted = predict(model, test.data)

print("\n Reporte")
print(metrics.classification_report(test.target, predicted, target_names=test.target_names, zero_division=0))

accuracy = metrics.accuracy_score(test.target, predicted)
precision_macro, recall_macro, f1_macro = evaluate_metrics(test.target, predicted, average='macro')
precision_weighted, recall_weighted, f1_weighted = evaluate_metrics(test.target, predicted, average='weighted')

# Evaluar métricas
precision, recall, f1 = evaluate_metrics(test.target, predicted, average='macro')
predicted = model.predict(test.data)

print("Métricas de Evaluacion")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Accuracy (Exactitud): {accuracy:.4f}")
print(f"Macro Avg -> Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
print(f"Weighted Avg -> Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
# Evaluar
print(metrics.classification_report(test.target, predicted, target_names=test.target_names))
