# News-Category-Classifier

This is a web application that classifies news articles into categories using Flask and a simple Linear Support Vector Classification Model.

![](https://github.com/matomoniwano/News-Category-Classifier/blob/master/image/news-category-classifier.png?raw=true)

## Prerequisites

* Flask
* Pickle

## Usage

To run the app, clone the repository and navigate to the root directory. Then, run the following command:

```bash
python app.py
```

The app will be hosted at [https://news-category-classifier-1.matomoniwano.repl.co/](https://news-category-classifier-1.matomoniwano.repl.co/).

To classify a news article, enter the text in the text box and click the "submit" button. The app will return the predicted category for the given text.

## Customization
To train the classifier with your own data, you can modify the **NLP_categorizer** function in the **app.py** file. The function loads in the pre-trained model using pickle and applies it to the input text to predict the category.

## Credits
Built by Matomo Niwano.
