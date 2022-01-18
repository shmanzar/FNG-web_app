from flask import Flask
from flask import request
from flask import render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route("/")
def index():
    headline = request.args.get('headline','')
    print(headline)
    result = "edgecase"
    prob = ""
    if headline:
        headline = [headline]
        prob, result = jeanie(headline)
        print (prob)
        print (result)
   
    return render_template(
        "home.html",
        result = result,
        prob = prob
    )
@app.route("/type")
def tempPage():
    return render_template(
        "typing.html"
    )

def fahrenheit_from(celsius):
    # -- snip --
    """Convert Celsius to Fahrenheit degrees."""
    try:
        fahrenheit = float(celsius) * 9 / 5 + 32
        fahrenheit = round(fahrenheit, 3)  # Round to three decimal places
        return str(fahrenheit)
    except ValueError:
        return "invalid input"
    
def jeanie(headline):

    filename_model = './finalized_model.sav'
    loaded_model = pickle.load(open(filename_model, 'rb'))

    filename_cv = './count_vectorizer.sav'
    cv = pickle.load(open(filename_cv, 'rb')) 

    # print(loaded_model)
    # print(cv)

    transform_input = cv.transform(headline)
    # print(transform_input)

    predicted_outcome = loaded_model.predict(transform_input)
    predicted_proba = loaded_model.predict_proba(transform_input)
    
    if predicted_outcome[0]:
        return((predicted_proba[0][1]*100), predicted_outcome[0])
        print(predicted_outcome)
        # print(predicted_proba[0][1]*100)
    else:
        return((predicted_proba[0][0]*100), predicted_outcome[0])
        print(predicted_outcome)
        # print(predicted_proba[0][0]*100)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)