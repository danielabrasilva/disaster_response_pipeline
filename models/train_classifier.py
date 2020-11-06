import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
nltk.download(['punkt', 'stopwords', 'wordnet']) # for word_tokenize, stopwords and lemmatizer, respectively
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """ Load data from a database.
    
    Args:
    database_filepath: str. Filepath of the database.
    
    Returns:
    X: numpy.array. Array containing message inputs.
    y: numpy.array. Array containing the message classifications.
    category_names: list. List with the classification labels.    
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('project_data', con=engine)
    X = df['message'].values
    y = df.iloc[:,4:].values
    category_names = df.iloc[:,4:].columns

    return X, y, category_names


def tokenize(text):
    """ Transform text string in a token list.

    Args:
    text: str. The text to tokenize.
    stop_words: bool. If is true, remove the stop words
    lemmatize (bool). If is true, lemmatize the tokens.

    Returns:
    tokens: list. Return a list of tokens from the text.

    """
    # Normalize text
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-0]'," ", text)


    #Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    tokens = [w for w in tokens if w not in stopwords.words("english")]

    # Reduce words to their root form
    tokens = [WordNetLemmatizer().lemmatize(w, pos='v') for w in tokens]


    return tokens


def build_model():
    """ Build model with NLP and ML Pipeline with a GridSearchCV sklearn model. 
    
    Returns:
    model: A model to be fitted with the training data.
    """
    
    
    # Instantiate pipeline
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(random_state=42)))
    ])
        
    # GridSearch algorithm    
    parameters= {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [100, 200],
    }
    
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model



def evaluate_model(model, X_test, Y_test, category_names):
    """ Print a text report showing the main classification metrics.
    
    Args:
    model: the trained model.
    X_test: numpy array. Array containing  tested inputs.
    Y_test: numpy array. Array containing tested outputs.
    category_names: list. List with category labels.
    """
    y_preds = model.predict(X_test)

    for i,cat in enumerate(category_names):
        classification = classification_report(Y_test[:,i], y_preds[:,i])
        print(cat+':\n')
        print(classification+'\n')


def save_model(model, model_filepath):
    "Save the model in a pickle file."
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
        print(model.best_params_)

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
