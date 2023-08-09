from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the pre-trained vectorizer and classifier
vectorizer = joblib.load('vectorizer.pkl')
classifier1 = joblib.load('classifier.pkl')

# Define prediction function
def make_prediction(text):
    text_bow = vectorizer.transform([text])
    prediction = classifier1.predict(text_bow)
    
    labels = {
        0: "hate-speech",
        1: "offensive-speech",
        2: "neither"
    }
    
    return labels[prediction[0]]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        text_input = request.form['text_input']
        if text_input:
            prediction = make_prediction(text_input)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    from gunicorn.app.wsgiapp import WSGIApplication
    app_wsgi = WSGIApplication()
    app_wsgi.app_uri = 'app:app'  # Assuming your Flask app is named 'app'
    app_wsgi.run()
