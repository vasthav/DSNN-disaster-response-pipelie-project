import re
import ssl
import sys

import nltk
# import libraries
import pandas as pd
from sqlalchemy import create_engine

# running nltk download gave ssl error.
# Resolution reference: https://github.com/gunthercox/ChatterBot/issues/930#issuecomment-322111087
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download(['punkt', 'wordnet', 'stopwords'])

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.model_selection import train_test_split


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('cleaned_data', engine)

    # Clean dataframe remove na values
    df.dropna(axis='index', inplace=True)

    # Preparing data
    X = df.message.values
    t = df.drop(columns=['id', 'message', 'original', 'genre']).astype(int)

    X = df['message']
    Y = t

    return X, Y, list(Y.columns)


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)

    stop_word = stopwords.words("english")
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    lemmed = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_word]

    return lemmed


def build_model():
    from sklearn.multioutput import MultiOutputClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10, n_jobs=-1)))
    ])

    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0),
        'tfidf__use_idf': (True, False),
        'vect__max_features': (None, 5000),
        'clf__estimator__n_estimators': [10, 20]
    }

    # Grid Search best parameters after running with above parameters.
    # parameters = {
    #     'vect__max_features': (None, 5000),
    #     'vect__ngram_range': (1, 2),
    #     'vect__max_df': (1.0),
    #     'tfidf__use_idf': (True),
    #     'clf__estimator__n_estimators': [20],
    # }

    # Testing parameters
    # parameters = {
    #     'vect__ngram_range': [(1, 1)],
    #     'clf__estimator__n_estimators': [10, 20],
    # }

    cv = GridSearchCV(estimator=pipeline, param_grid=parameters, scoring='f1_macro', cv=None, n_jobs=-1, verbose=10)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # references:
    # https://stackoverflow.com/questions/38697982/python-scikit-learn-multi-class-multi-label-performance-metrics
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html?highlight=report#sklearn.metrics.classification_report

    # predict on test data
    predicted = model.predict(X_test)

    from sklearn.metrics import classification_report

    y_pred_pd = pd.DataFrame(predicted, columns=Y_test.columns)
    for column in Y_test.columns:
        print(f'Class: {column}\n')
        print(classification_report(Y_test[column], y_pred_pd[column]))

    # Finding the Accuracy
    accuracy = (predicted == Y_test).mean()
    print(f'accuracy = {accuracy.mean()} \n')


def save_model(model, model_filepath):
    import pickle
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier_simple.pkl')


if __name__ == '__main__':
    main()
