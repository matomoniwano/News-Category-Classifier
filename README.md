# News-Category-Classifier

This is a web application that classifies news articles into categories using Flask and a simple Linear Support Vector Classification Model.

## Prerequisites

* Flask
* Pickle

## Usage

To run the app, clone the repository and navigate to the root directory. Then, run the following command:

```bash
python app.py
```

The app will be hosted at http://127.0.0.1:5000/.

To classify a news article, enter the text in the text box and click the "submit" button. The app will return the predicted category for the given text.

## Customization
To train the classifier with your own data, you can modify the **NLP_categorizer** function in the **app.py** file. The function loads in the pre-trained model using pickle and applies it to the input text to predict the category.

## Credits
Built by Matomo Niwano.
