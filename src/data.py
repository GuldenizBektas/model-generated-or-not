import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import plotly as py
import plotly.graph_objs as go
from plotly import tools
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from plotly.offline import iplot,plot
from plotly.offline import init_notebook_mode, iplot,plot

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import auc, roc_auc_score, roc_curve, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

### import data
path = "data/train_essays.csv"
absolute_path = os.path.join(os.getcwd(), path)
data = pd.read_csv(absolute_path)

# external data
ex_1 = pd.read_csv(os.path.join(os.getcwd(), "data/train_drcat_04.csv"))[["text", "label", "source"]]
ex_2 = pd.read_csv(os.path.join(os.getcwd(), "data/argugpt.csv"))[["text", "model"]]

data.rename(columns={"generated": "label"}, inplace=True)
data = data[["text", "label"]]
ex_1 = ex_1[["text", "label"]]
ex_2["label"] = 1
ex_2 = ex_2[["text", "label"]]

df = pd.concat([data, ex_1, ex_2])

# shuffle
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(df.text, df.label, test_size=.2, random_state=42)

vectorizer = CountVectorizer()
train_cv = vectorizer.fit_transform(X_train)
print(train_cv.shape)

tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_cv)
print(train_tfidf.shape)

test_cv = vectorizer.transform(X_test)
test_tfids = tfidf.transform(test_cv)

#### Machine learning

def classification_models(model, save_path=None):

    print("********* Training Model **************")
    y_pred=model.fit(train_tfidf,y_train).predict(test_tfids)
    print("********* Training done 🎉 ************")
    accuracy=accuracy_score(y_test, y_pred)
    f1=f1_score(y_test, y_pred, average="weighted")
    precision=precision_score(y_test, y_pred, average="weighted")
    recall=recall_score(y_test, y_pred, average="weighted")
    
    results=pd.DataFrame({"Values":[accuracy,f1,precision,recall],
                         "Metrics":["Accuracy","F1","Precision","Recall"]})
    
    # Visualize Results:
    fig=make_subplots(rows=1,cols=1)
    fig.add_trace(go.Bar(x=[round(i,5) for i in results["Values"]],
                        y=results["Metrics"],
                        text=[round(i,5) for i in results["Values"]],orientation="h",textposition="inside",name="Values",
                        marker=dict(color=["indianred","firebrick","palegreen","skyblue","plum"],line_color="beige",line_width=1.5)),row=1,col=1)
    fig.update_layout(title={'text': model.__class__.__name__ ,
                             'y':0.9,
                             'x':0.5,
                             'xanchor': 'center',
                             'yanchor': 'top'},
                      template='plotly_white')
    fig.update_xaxes(range=[0,1], row = 1, col = 1)

    if save_path:
        # Save the figure as an HTML file
        fig.write_image(save_path)
        print(f"Figure saved to {save_path}")
    else:
        # Show the figure if save_path is not provided
        fig.show()


def conf_matrix(y, y_pred, title):
    fig, ax =plt.subplots(figsize=(5,5))
    labels=["0", "1"]
    ax=sns.heatmap(confusion_matrix(y, y_pred), annot=True, cmap="Blues", fmt='g', cbar=False, annot_kws={"size":25})
    plt.title(title, fontsize=20)
    ax.xaxis.set_ticklabels(labels, fontsize=17) 
    ax.yaxis.set_ticklabels(labels, fontsize=17)
    ax.set_ylabel('Test', fontsize=20)
    ax.set_xlabel('Predicted', fontsize=20)
    plt.show()

my_models= [
    
    LogisticRegression(),
    KNeighborsClassifier(),
    RandomForestClassifier(),
    XGBClassifier(),
    DecisionTreeClassifier(),
    SVC()


]

for model in my_models:
    classification_models(model, save_path="/Users/guldenizbektas/Documents/LLM_generated_text/results/"+str(model)+".png")
