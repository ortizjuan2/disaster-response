import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
# Pipeline
from sklearn.pipeline import Pipeline
import joblib

# download words for nltk package
nltk.download(['punkt', 'stopwords', 'wordnet'])
stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()


def load_data(database_filepath):
    """ load table from sqlite database

    input:
        database_filepath: str, path to messages sqlite database

    return:
        X: dataset with data features
        Y: dataset with target variables
        category_names: list of categories in the data
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('categories', engine)
    X = df['message']
    # target variables are the 36 categories
    Y = df.iloc[:, 4:]
    category_names = list(Y.columns)
    return (X, Y, category_names)


def tokenize(text):
    """ Extract words from message, lower text, remove punctuation and stopwords

    input:
        text: str containing the message
    
    output:
        tokens: list of words
    """
    # remove non alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text.lower())
    # tokenize text
    tokens = word_tokenize(text)
    # lemmatize and remove stop words
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return tokens


def build_model():
    """ Build a model using pipeline and grid search
    """
    parameters = {"clf__splitter": ["best", "random"],
            # "clf__max_depth" : [None,10],
            # "clf__min_samples_leaf":[1,2],
            # "clf__min_weight_fraction_leaf":[0.,0.1],
            "clf__max_features": ["log2", "sqrt", None],
            # "clf__max_leaf_nodes":[None,20,50]
        }

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', DecisionTreeRegressor()), ])

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """ Evaluate the model to display f1 and recall scores

    input:
        model: Grid search pipeline
        X_test: dataset with test features data
        Y_test: dataset with test target data
        category_names: list of categories
    """
    Y_pred = model.predict(X_test)
    f1 = [f1_score(Y_test.iloc[:, i].values, Y_pred[:, i].astype(int)) for i in range(Y_test.shape[1])]
    recall = [recall_score(Y_test.iloc[:, i].values, Y_pred[:, i].astype(int)) for i in range(Y_test.shape[1])]
    score = pd.DataFrame({
            'category_names':category_names,
            'f1_score':f1,
            'recall_score':recall,
        })
    print(score)


def save_model(model, model_filepath):
    """ Save trained model

    input:
        model: Grid search pipeline
        model_filepath: str, model file path
    """
    joblib.dump(model.best_estimator_, model_filepath, compress=1)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
