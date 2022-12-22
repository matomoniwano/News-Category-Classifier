from flask import Flask, render_template, request, flash
import pickle

def predict_category(text, classifier, tfidf_vectorizer):
    result = classifier.predict(tfidf_vectorizer.transform([text]))
    return(result[0])
    

def NLP_categorizer(txt):

    # Use pickle to load in the pre-trained model.
    with open('./classifier/vector.pkl', 'rb') as file:
        tfidf_vectorizer = pickle.load(file)
        
    with open('./classifier/class.pkl', 'rb') as file:
        classifier = pickle.load(file)

    return predict_category(txt, classifier, tfidf_vectorizer)

app = Flask(__name__)
app.secret_key = "momo"

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/start")
def start():
    flash("↓ Enter text here ↓")
    return render_template("index.html")


@app.route("/result", methods=['POST', 'GET'])
def result():
    flash("↓ Enter text here ↓")
    if len(request.form['input']) < 1:
        nottyped = "Please Enter Text in the text box!"
        return render_template("index.html", nottyped = nottyped)
    else:
        output = NLP_categorizer(request.form['input'])
        return render_template("index.html", output = output)

if __name__ == '__main__':
   app.run(debug = True)