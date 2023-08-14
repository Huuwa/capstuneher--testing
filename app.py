import warnings
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Suppress inconsistent version warnings for scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")

# Load the pre-trained vectorizer and classifiers
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
bow_vectorizer = joblib.load('bow_vectorizer.pkl')
classifier_tfidf = joblib.load('tfidf_classifier.pkl')
classifier_bow = joblib.load('bow_classifier.pkl')

# Define prediction function
def make_predictions(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    text_bow = bow_vectorizer.transform([text])

    prediction_tfidf = classifier_tfidf.predict(text_tfidf)
    prediction_bow = classifier_bow.predict(text_bow)
    
    confidence_tfidf = classifier_tfidf.predict_proba(text_tfidf).max()
    confidence_bow = classifier_bow.predict_proba(text_bow).max()

    labels = {
        0: "hate-speech",
        1: "offensive-speech",
        2: "neither"
    }

    return {
        'tfidf': (labels[prediction_tfidf[0]], confidence_tfidf),
        'bow': (labels[prediction_bow[0]], confidence_bow)
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    best_prediction = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        if text_input:
            predictions = make_predictions(text_input)
            best_prediction = max(predictions, key=lambda k: predictions[k][1])

    return render_template('index.html', predictions=predictions, best_prediction=best_prediction)

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import WSGIApplication
    app_wsgi = WSGIApplication()
    app_wsgi.app_uri = 'app:app'  # Assuming your Flask app is named 'app'
    app_wsgi.run()
