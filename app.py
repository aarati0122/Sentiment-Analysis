
import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
# from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
# import nltk
# import contractions
# import re
# from nltk.tokenize.toktok import ToktokTokenizer
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import plot_confusion_matrix

import warnings
warnings.filterwarnings("ignore")


from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# import datetimet

vs = SentimentIntensityAnalyzer()

def main():
  st.title("SENTIMENT ANALYSIS")

  def load_data():
    df = pd.read_csv('https://raw.githubusercontent.com/aarati0122/Datasets/main/news.csv')
    filtrate_data = pd.read_csv('https://raw.githubusercontent.com/aarati0122/Datasets/main/filtrate.csv')
    return df,filtrate_data
  
  def split(df):
    df = dataframe(df)
    x = df.news_article
    y = df.sentiment
    xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0)
    return xtrain, xtest, ytrain, ytest
  def dataframe(df):
    df.loc[df['compound'] > -0.05 , 'sentiment'] = 'neutral' 
    df.loc[df['compound'] <= -0.05, 'sentiment'] = 'negative' 
    df.loc[df['compound'] >= 0.05, 'sentiment'] = 'positive' 
    df = df[["news_headline","news_article","sentiment","compound"]]
    return df

  def piechart(df):
    positive = len(df[df["sentiment"]=="positive"])
    negative = len(df[df["sentiment"]=="negative"])
    neutral = len(df[df["sentiment"]=="neutral"])
    pie_sentiment = df.sentiment.value_counts(normalize=True)
    plt.figure(figsize=(2,2),dpi=10)
    labels = 'Positive','Negative',"Neutral"
    colors = ['yellowgreen', 'red','gold']
    plt.pie(pie_sentiment, colors=colors,labels=labels,autopct='%1.1f%%')
    plt.style.use('default')
    plt.axis('equal')
    sizes = [positive, neutral, negative]
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

  def unigram_cv():
    x_train, x_test, y_train, y_test = split(filtrate)
    cv1 = CountVectorizer(stop_words='english')

    x_train_cv1 = cv1.fit_transform(x_train)
    x_test_cv1  = cv1.transform(x_test)

    ug_cv = pd.DataFrame(x_train_cv1.toarray(), columns=cv1.get_feature_names()).sample(10)
    return x_train_cv1,x_test_cv1

  def bigram_cv():
    x_train, x_test, y_train, y_test = split(filtrate)
    cv2 = CountVectorizer(ngram_range=(1,2), binary=True, stop_words='english')
    x_train_cv2 = cv2.fit_transform(x_train)
    x_test_cv2  = cv2.transform(x_test)
    big_cv = pd.DataFrame(x_train_cv2.toarray(), columns=cv2.get_feature_names()).head()
    return x_train_cv2,x_test_cv2

  def logisticReg_KNN1():
    x_train_cv1,x_test_cv1 =unigram_cv()
    lr = LogisticRegression()
      # Train the first model
    lr.fit(x_train_cv1, y_train)
    y_pred_cv1 = lr.predict(x_test_cv1)
    confusion = confusion_matrix(y_test, y_pred_cv1)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                  xticklabels=['negative', 'neutral','positive'],
                  yticklabels=['negative', 'neutral','positive'],
                  fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    accuracy = accuracy_score(y_test, y_pred_cv1)
    f1 = f1_score(y_test, y_pred_cv1,average='weighted')
    precision = precision_score(y_test, y_pred_cv1,average='weighted')
    recall = recall_score(y_test, y_pred_cv1,average='weighted')
    cm1 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm1
  

  def logisticReg_KNN2():
     # Train the second model
    x_train_cv2,x_test_cv2 =bigram_cv()
    lr = LogisticRegression()
    lr.fit(x_train_cv2, y_train)
    y_pred_cv2 = lr.predict(x_test_cv2)


      # Print confusion matrix for kNN
    confusion = confusion_matrix(y_test, y_pred_cv2)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                  xticklabels=['negative', 'neutral','positive'],
                  yticklabels=['negative', 'neutral','positive'],
                  fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

      # Print Sklearn Metrices
    accuracy = accuracy_score(y_test, y_pred_cv2)
    f1 = f1_score(y_test, y_pred_cv2,average='weighted')
    precision = precision_score(y_test, y_pred_cv2,average='weighted')
    recall = recall_score(y_test, y_pred_cv2,average='weighted')
    cm2 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm2
    
  def naive_bayes1():
    # Fit the first Naive Bayes model
    x_train_cv1,x_test_cv1 =unigram_cv()
    mnb = MultinomialNB()
    mnb.fit(x_train_cv1, y_train)
    y_pred_cv1_nb = mnb.predict(x_test_cv1)

    # Print confusion matrix for kNN
    confusion = confusion_matrix(y_test, y_pred_cv1_nb)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                xticklabels=['negative', 'neutral','positive'],
                yticklabels=['negative', 'neutral','positive'],
                fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    # Print Sklearn Metrices
    accuracy = accuracy_score(y_test, y_pred_cv1_nb)
    f1 = f1_score(y_test, y_pred_cv1_nb,average='weighted')
    precision = precision_score(y_test, y_pred_cv1_nb,average='weighted')
    recall = recall_score(y_test, y_pred_cv1_nb,average='weighted')
    cm3 = [accuracy, precision, recall, f1]
    # st.write("Accuracy: {:.2%}".format(accuracy))
    # st.write("Precision: {:.2%}".format(precision))
    # st.write("Recall: {:.2%}".format(recall))
    # st.write("F1 Score: {:.2%}".format(f1))
    return cm3

  def naive_bayes2():
    # Fit the second Naive Bayes model
    x_train_cv2,x_test_cv2 =bigram_cv()
    mnb = MultinomialNB()
    mnb.fit(x_train_cv2, y_train)

    y_pred_cv2_nb = mnb.predict(x_test_cv2)

    confusion = confusion_matrix(y_test, y_pred_cv2_nb)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                  xticklabels=['negative', 'neutral','positive'],
                  yticklabels=['negative', 'neutral','positive'],
                  fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    accuracy = accuracy_score(y_test, y_pred_cv2_nb)
    f1 = f1_score(y_test, y_pred_cv2_nb,average='weighted')
    precision = precision_score(y_test, y_pred_cv2_nb,average='weighted')
    recall = recall_score(y_test, y_pred_cv2_nb,average='weighted')
    cm4 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm4

  def KNeighbors1():
    # Fit the first KNN model
    x_train_cv1,x_test_cv1 =unigram_cv()
    knn = KNeighborsClassifier(3)
    knn.fit(x_train_cv1, y_train)

    y_pred_cv1_knn = knn.predict(x_test_cv1)

    confusion = confusion_matrix(y_test, y_pred_cv1_knn)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                xticklabels=['negative', 'neutral','positive'],
                yticklabels=['negative', 'neutral','positive'],
                fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    accuracy = accuracy_score(y_test, y_pred_cv1_knn)
    f1 = f1_score(y_test, y_pred_cv1_knn,average='weighted')
    precision = precision_score(y_test, y_pred_cv1_knn,average='weighted')
    recall = recall_score(y_test, y_pred_cv1_knn,average='weighted')
    cm5 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm5

  def KNeighbors2():
    # Fit the second KNN model
    x_train_cv2,x_test_cv2 =bigram_cv()
    knn = KNeighborsClassifier(3)
    knn.fit(x_train_cv2, y_train)

    y_pred_cv2_knn = knn.predict(x_test_cv2)

    confusion = confusion_matrix(y_test, y_pred_cv2_knn)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                xticklabels=['negative', 'neutral','positive'],
                yticklabels=['negative', 'neutral','positive'],
                fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    accuracy = accuracy_score(y_test, y_pred_cv2_knn)
    f1 = f1_score(y_test, y_pred_cv2_knn,average='weighted')
    precision = precision_score(y_test, y_pred_cv2_knn,average='weighted')
    recall = recall_score(y_test, y_pred_cv2_knn,average='weighted')
    cm6 = [accuracy, precision, recall, f1]
    print("Accuracy: {:.2%}".format(accuracy))
    print("Precision: {:.2%}".format(precision))
    print("Recall: {:.2%}".format(recall))
    print("F1 Score: {:.2%}".format(f1))
    return cm6

  def DecisionTree1():
    x_train_cv1,x_test_cv1 =unigram_cv()
    dt = DecisionTreeClassifier()
    dt.fit(x_train_cv1, y_train)

    y_pred_cv1_dt = dt.predict(x_test_cv1)

    confusion = confusion_matrix(y_test, y_pred_cv1_dt)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                xticklabels=['negative', 'neutral','positive'],
                yticklabels=['negative', 'neutral','positive'],
                fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()

    accuracy = accuracy_score(y_test, y_pred_cv1_dt)
    f1 = f1_score(y_test, y_pred_cv1_dt,average='weighted')
    precision = precision_score(y_test, y_pred_cv1_dt,average='weighted')
    recall = recall_score(y_test, y_pred_cv1_dt,average='weighted')
    cm7 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm7

  def DecisionTree2():
    x_train_cv2,x_test_cv2 =bigram_cv()
    dt = DecisionTreeClassifier()
    dt.fit(x_train_cv2, y_train)

    y_pred_cv2_dt = dt.predict(x_test_cv2)

    confusion = confusion_matrix(y_test, y_pred_cv2_dt)
    plt.figure(dpi=100)
    sns.heatmap(confusion, cmap=plt.cm.Blues, annot=True, square=True,
                xticklabels=['negative', 'neutral','positive'],
                yticklabels=['negative', 'neutral','positive'],
                fmt='d', annot_kws={'fontsize':20})
    plt.xticks(rotation=0)
    plt.title('Confusion matrix')
    plt.xlabel('Predicted Sentiments')
    plt.ylabel('Actual Sentiments')
    st.pyplot()

    accuracy = accuracy_score(y_test, y_pred_cv2_dt)
    f1 = f1_score(y_test, y_pred_cv2_dt,average='weighted')
    precision = precision_score(y_test, y_pred_cv2_dt,average='weighted')
    recall = recall_score(y_test, y_pred_cv2_dt,average='weighted')
    cm8 = [accuracy, precision, recall, f1]
    st.write("Accuracy: {:.2%}".format(accuracy))
    st.write("Precision: {:.2%}".format(precision))
    st.write("Recall: {:.2%}".format(recall))
    st.write("F1 Score: {:.2%}".format(f1))
    return cm8

  def logs():
    log_file = st.selectbox("What logs do you want to see",("Select the Options","Log1","Log2","Log3","Log4"))
    if "Log1" in log_file:
      log1 = pd.read_csv("https://raw.githubusercontent.com/aarati0122/Sentiment-Analysis/main/log1.csv")
      st.write(log1)
    if "Log2" in log_file:
      log2 = pd.read_csv("https://raw.githubusercontent.com/aarati0122/Sentiment-Analysis/main/log2.csv")
      st.write(log2)
    if "Log3" in log_file:
      log3 = pd.read_csv("https://raw.githubusercontent.com/aarati0122/Sentiment-Analysis/main/log3.csv")
      st.write(log3)
    if "Log4" in log_file:
      log4 = pd.read_csv("https://raw.githubusercontent.com/aarati0122/Sentiment-Analysis/main/log4.csv")
      st.write(log4)
  
    


  data,filtrate = load_data()
  x_train, x_test, y_train, y_test = split(filtrate)


  dataset = st.sidebar.selectbox("Dataset",("select the option","Without Filtrate Dataset",
                                            "With Filtrate Dataset","Sentiments","Logs of Sentiments"))
  if dataset  == "Without Filtrate Dataset":
    st.subheader("Dataset from web scraping")
    st.write(data)
  
  if dataset  == "With Filtrate Dataset":
    st.subheader("Filtrated Data")
    st.write(filtrate)
  
  if dataset  == "Sentiments":
    data_fil = st.sidebar.checkbox("Sentiments")
    sentiment = dataframe(filtrate)
    if(data_fil):
      st.subheader("Adding the sentiments")
      st.write(sentiment)
     
    ch = st.sidebar.checkbox("Pie chart of Sentiment Analysis")
    if(ch):
      st.subheader('Sentiment Rate for web scrapng data')
      chart = piechart(sentiment)
      st.write("Total records: ",len(sentiment))
      st.write("Train records: ",len(x_train))
      st.write("Test records : ",len(x_test))
      st.write(chart)

  if dataset == "Logs of Sentiments":
      logs()

  regression = st.sidebar.selectbox("Regression",("select regression","Logistic Regression","MultinomialNB","KNeighbors Classifier"
                  ,"DecisionTree Classifier","Normalized confusion matrix"))

  if regression == "Logistic Regression":
    # Create a logistic regression model to use
    cv1 = st.sidebar.checkbox("Countvector for 1st train Model for Logistic Regression")
    if(cv1):
      logisticReg_KNN1()

    cv2 = st.sidebar.checkbox("Countvector for 2nd train Model for Logistic Regression")
    if(cv2):
      logisticReg_KNN2()
     
    
    campare = st.sidebar.checkbox('Compile all of the error metrics into a dataframe for comparison')
    if(campare):
    # Compile all of the error metrics into a dataframe for comparison
      cm1 = logisticReg_KNN1()
      cm2 =  logisticReg_KNN2()
      results = pd.DataFrame(list(zip(cm1, cm2)))
      results = results.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
      results.columns = ['LR1-CV', 'LR2-CV-Ngr']
      st.write(round(results,3))

  if regression == "MultinomialNB":
    cv1 = st.sidebar.checkbox("MultinomialNB 1")
    if(cv1):
      naive_bayes1()
    
    cv2 = st.sidebar.checkbox("MultinomialNB 2")
    if(cv2):
      naive_bayes2()

    campare = st.sidebar.checkbox("Compile all of the error metrics into a dataframe for comparison")
    if(campare):
      cm3 = naive_bayes1()
      cm4 = naive_bayes2()
      results_nb = pd.DataFrame(list(zip(cm3, cm4)))
      results_nb = results_nb.set_index([['Accuracy', 'Precision', 'Recall', 'F1 Score']])
      results_nb.columns = ['NB1-CV', 'NB2-CV-Ngr']
      results_nb
      results = pd.concat([results, results_nb], axis=1)
      st.write(results)
  if regression == "KNeighbors Classifier":
    cv1 = st.sidebar.checkbox("KNeighbors Classifier 1")
    if(cv1):
      KNeighbors1()
    cv2 = st.sidebar.checkbox("KNeighbors Classifier 2")
    if(cv2):
      KNeighbors1()
  
  if regression == "DecisionTree Classifier":
    cv1 = st.sidebar.checkbox("DecisionTree Classifier 1")
    if(cv1):
      DecisionTree1()
    cv2 = st.sidebar.checkbox("DecisionTree Classifier 2")
    if(cv2):
      DecisionTree2()
  if regression == "Normalized confusion matrix":
    x_train_cv2,x_test_cv2 =bigram_cv()
    classifier = LogisticRegression().fit(x_train_cv2, y_train)

    np.set_printoptions(precision=2)
    plt.figure(figsize=(15,10),dpi=80),
    plt.style.use('default')

    # Plot non-normalized confusion matrix
    titles_options = [("Confusion matrix", None),
                      ("Normalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, x_test_cv2, y_test,
                                    display_labels=['negative', 'neutral','positive'],
                                    cmap=plt.cm.Blues,
                                    normalize=normalize)
        disp.ax_.set_title(title)

    st.pyplot()
if __name__ == '__main__':
  main()
