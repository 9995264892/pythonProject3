from flask import Flask, render_template, url_for, request, app
import pandas as pd
import xlrd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 'dataframe 1'
    df = pd.read_csv('/Users/avdeeviv/PycharmProjects/pythonProject3/Spam/001.csv', encoding="latin-1")
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df = df.rename(columns={'v1': 'label', 'v2': 'text'})
    df = df.replace(['ham', 'spam'], [0, 1])
    # 'dataframe 2'
    df1 = pd.read_csv('/Users/avdeeviv/PycharmProjects/pythonProject3/Spam/002.csv', encoding="latin-1")
    df1 = df1.drop(["subject"], axis=1)  # 'удаление столбца subject'
    df1 = df1.rename(columns={'message': 'text'})
    pd.concat([df, df1])
    X = df['text']
    y = df['label']
    cv = CountVectorizer()
    X = cv.fit_transform(X)  # Fit the Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    #   'простой вероятностный классификатор, основанный на применении теоремы Байеса со строгими предположениями о независимости'
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    clf.score(X_test, y_test)
    joblib.dump(clf, 'NB_spam_model.pkl')
    #   'y_pred = clf.predict(X_test)'
    #   'print(classification_report(y_test, y_pred))'
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction=my_prediction)


if __name__ == '__main__':
    app.run(debug=True)
