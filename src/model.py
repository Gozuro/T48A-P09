from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

def create_model():
    pass
    return make_pipeline(TfidfVectorizer(), MultinomialNB())
def train_model(model, data, target):
    pass
    model.fit(data, target)
    return model

def predict(model, data):
    pass
    return model.predict(data)
