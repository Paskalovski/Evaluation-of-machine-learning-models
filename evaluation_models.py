import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## Data visualization 
df = pd.read_csv("fraud_data.csv")
df.head()
df.info()

fraud = df['Class']
fraud.head()

# Fraud data percentage
def fraud_percent():
    fraud = df['Class']
    return sum(fraud)/len(fraud)
fraud_percent()

## Data preprocessing 
# Defining the training and testing data 
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


## Using the dummy classifier to classify the dataset and calculating the accuracy and recall score

def dummy_classifier():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    dummy= DummyClassifier(strategy = 'most_frequent').fit(X_train,y_train)
    y_predict = dummy.predict(X_test)
    accuracy_score = dummy.score(X_test,y_test)
    recall_score = recall_score(y_test,y_predict)
    return (accuracy_score,recall_score)
dummy_classifier()


## Using the SVC classifier to classify the dataset and calculating the accuracy, recall, and precision score

def svc_classifier():
    from sklearn.metrics import recall_score, precision_score, accuracy_score
    from sklearn.svm import SVC

    svm = SVC().fit(X_train, y_train)
    svm_predicted_mc = svm.predict(X_test)

    return accuracy_score(y_test, svm_predicted_mc), recall_score(y_test, svm_predicted_mc), precision_score(y_test, svm_predicted_mc)
svc_classifier()


## Using the SVC classifier (with tunned parameters) to classify the dataset and calculating the confusion matrix

def confussion_matrix():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC

    svm = SVC(gamma=1e-07, C=1e9).fit(X_train, y_train)
    svm_predicted_mc = svm.decision_function(X_test) > -220
    confusion_mc = confusion_matrix(y_test, svm_predicted_mc)

    return confusion_mc
confussion_matrix()


## Using the Logistic regression classifier to classify the dataset
## Analyzing the precision recall and ROC curve

def log_regression():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve, auc

    lr = LogisticRegression().fit(X_train, y_train)
    lr_predicted = lr.predict(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, lr_predicted)
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_predicted)


    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()

    roc_auc_lr = auc(fpr_lr, tpr_lr)
    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC Curve', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    
    return plt.show()

log_regression()


## Performing a grid search over the defined parameters for a logistic regression and using recall for scoring and the default 3-fold cross validation
def grid_search():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression().fit(X_train, y_train)
    grid_values = {'penalty': ['l1', 'l2'], 'C': [0.01, 0.1, 1, 10, 100]}
    grid_clf_rec = GridSearchCV(lr, param_grid=grid_values, scoring='recall')
    grid_clf_rec.fit(X_train, y_train)

    return np.array([grid_clf_rec.cv_results_['mean_test_score'][x:x + 2] for x in
                     range(0, len(grid_clf_rec.cv_results_['mean_test_score']), 2)])


# Graphical presentation 
def GridSearch_Heatmap(scores):
   plt.figure()
   sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10, 100])
   plt.yticks(rotation=0);
   plt.show()

GridSearch_Heatmap(grid_search())
