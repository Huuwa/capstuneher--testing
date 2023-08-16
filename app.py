import warnings
from flask import Flask, render_template, request
import joblib
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import pickle


app = Flask(__name__, static_folder="static")

# Suppress inconsistent version warnings for scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, message=".*InconsistentVersionWarning.*")

# Load the pre-trained vectorizers and classifiers
tfidf_vectorizer = joblib.load('tfidf_vectorizer_updated.pkl')
bow_vectorizer = joblib.load('bow_vectorizer_updated.pkl')
classifier_tfidf = joblib.load('tfidf_classifier_updated.pkl')
classifier_bow = joblib.load('bow_classifier_updated.pkl')


lstm_model = load_model('lstm_model_with_dropout.h5')
gru_model = load_model('gru_model_word_tokenization.h5')


with open('lstm_tokenizer.pkl', 'rb') as f:
    tokenizer_lstm = pickle.load(f)

with open('gru_tokenizer.pkl', 'rb') as f:
    tokenizer_gru = pickle.load(f)

# Initialize Tokenizer for Deep Learning models
n_most_common_words = 10000
max_sequence_length = 77  # Update this with the appropriate sequence length



# Define prediction function for TF-IDF model
def make_tfidf_predictions(text):
    text_tfidf = tfidf_vectorizer.transform([text])
    prediction_tfidf = classifier_tfidf.predict(text_tfidf)
    confidence_tfidf = classifier_tfidf.predict_proba(text_tfidf).max()
    label = ['hate-speech', 'offensive-speech', 'neither'][prediction_tfidf[0]]
    return label, confidence_tfidf

# Define prediction function for BoW model
def make_bow_predictions(text):
    text_bow = bow_vectorizer.transform([text])
    prediction_bow = classifier_bow.predict(text_bow)
    confidence_bow = classifier_bow.predict_proba(text_bow).max()
    label = ['hate-speech', 'offensive-speech', 'neither'][prediction_bow[0]]
    return label, confidence_bow

def make_lstm_predictions(text):
    sequence = tokenizer_lstm.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = lstm_model.predict(sequence)
    label = ['hate-speech', 'offensive-speech', 'neither'][np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence

def make_gru_predictions(text):
    sequence = tokenizer_gru.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    prediction = gru_model.predict(sequence)
    label = ['hate-speech', 'offensive-speech', 'neither'][np.argmax(prediction)]
    confidence = np.max(prediction)
    return label, confidence

def get_best_prediction_label(prediction):
    labels = {
        'tfidf': 'TF-IDF',
        'bow': 'BoW',
        'lstm': 'LSTM',
        'gru': 'GRU'
    }
    return labels.get(prediction, '')

def get_best_prediction_result(prediction):
    if prediction == 'hate-speech':
        return 'Hate-Speech'
    elif prediction == 'offensive-speech':
        return 'Offensive Speech'
    elif prediction == 'neither':
        return 'Neither'
    else:
        return 'Unknown'

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = None
    best_prediction = None
    
    if request.method == 'POST':
        text_input = request.form['text_input']
        
        if text_input:
            predictions = {
                'tfidf': make_tfidf_predictions(text_input),
                'bow': make_bow_predictions(text_input),
                'lstm': make_lstm_predictions(text_input),  # Add LSTM predictions
                'gru': make_gru_predictions(text_input)     # Add GRU predictions
            }
            best_prediction = max(predictions, key=lambda k: predictions[k][1])
            
    return render_template('index.html', predictions=predictions, best_prediction=best_prediction, get_best_prediction_label=get_best_prediction_label, get_best_prediction_result=get_best_prediction_result)

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import WSGIApplication
    app_wsgi = WSGIApplication()
    app_wsgi.app_uri = 'app:app'  # Assuming your Flask app is named 'app'
    app_wsgi.run()

