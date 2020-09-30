import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_score, recall_score, roc_auc_score, \
    roc_curve
import time

# set pycharm terminal wider than default
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 14)

PATH = '/home/tav/Desktop/Untitled Folder/data/'
dataset = pd.read_csv(f'{PATH}train.csv')
test_data = pd.read_csv(f'{PATH}test.csv')


# Function for fitting model, calculating metrics and adding to ROC plot
# global variables used as parameters. Improvement?
def create_model(model_type, X_train, y_train, X_test, y_test, roc_plot):
    model_name = type(model_type).__name__
    time_start = time.perf_counter()
    model = model_type.fit(X_train, y_train)
    time_elapsed = (time.perf_counter() - time_start)

    prob = model_type.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    auc = roc_auc_score(y_test, prob)
    fpr, tpr, _ = roc_curve(y_test, prob)

    print(f"{model_name} Metrics")
    print("Computation Time:%5.4f seconds" % time_elapsed)
    print("Accuracy: %.2f" % accuracy)
    print("Precision: %.2f" % precision)
    print("Recall: %.2f" % recall)
    print('AUC=%.3f' % auc, "\n")
    confusion_matrix(model, model_name)
    roc_plot.plot(fpr, tpr, marker='.', label=f'{model_name}')


# Plots confusion matrix for ML model
def confusion_matrix(model, model_name):
    plot_confusion_matrix(model, X_test, y_test, normalize='true')
    plt.title(f'{model_name} Confusion Matrix')

# No stand out data to extract from Ticket or Cabin that won't just correlate to Class/Fare
remove_cols = ['Ticket', 'Cabin']
numeric = ['Age', 'SibSp', 'Parch', 'Fare']  # histograms
categorical = ['Pclass', 'Sex', 'Embarked']  # valuecount()

for i in numeric:
    plt.hist(dataset[i])
    plt.title(i)
    plt.show()
for i in categorical:
    plt.bar(dataset[i].value_counts().index, dataset[i].value_counts())
    plt.title(i)
    plt.show()

titles = []
for person in dataset['Name']:
    titles.append(person.split(',')[1].split('.')[0])
titles = pd.DataFrame(titles)  # .value_counts()
dataset['Titles'] = titles
dataset = dataset.drop(columns=['Name']).drop(columns=remove_cols)

# Binary genders
map_gender = {'male': 1, 'female': 0}
dataset['Sex'] = dataset['Sex'].replace(map_gender)

# GROUP OTHER TITLES AS OTHER ->>> forgotten better method.
map_titles = {' Mr': 0, ' Miss': 1, ' Ms': 1, ' Mrs': 2, ' Master': 3, ' Don': 4, ' Rev': 4, ' Dr': 4, ' Mme': 4,
              ' Major': 4, ' Lady': 4, ' Sir': 4, ' Mlle': 4, ' Col': 4, ' Capt': 4, ' the Countess': 4,
              ' Jonkheer': 4, ' Dona': 4}
dataset['Titles'] = dataset['Titles'].replace(map_titles)
dataset['Titles'] = dataset['Titles'].astype(int)

map_emb = {'S': 0, 'C': 1, 'Q': 2}
dataset['Embarked'] = dataset['Embarked'].replace(map_emb)
# By far mode is 0, so fillna 0
dataset['Embarked'] = dataset['Embarked'].fillna(0).astype(int)

dataset['Fare-z'] = stats.zscore(dataset['Fare'])
# Silly arbitrary number, but removes extreme extremes: 7??
dataset = dataset.loc[dataset['Fare-z'].abs() <= 7]
dataset = dataset.reset_index().drop(columns=['Fare-z', 'index'])

# NORMALISE
# Create x, where x the 'scores' column's values as floats
min_max_scaler = preprocessing.MinMaxScaler()
fare_scaled = min_max_scaler.fit_transform(dataset[['Fare']].values.astype(float))
fare_normalised = pd.DataFrame(fare_scaled)
dataset[['Fare']] = fare_normalised

dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
age_scaled = min_max_scaler.fit_transform(dataset[['Age']].values.astype(float))
age_normalised = pd.DataFrame(age_scaled)
dataset[['Age']] = age_normalised

# SAMPLING
# https://elitedatascience.com/imbalanced-classes #
# Detect correlation between features
df_majority = dataset[dataset['Survived'] == 0]
df_minority = dataset[dataset['Survived'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(dataset[dataset['Survived'] == 0]),
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled = df_upsampled.sample(frac=1, random_state=30).reset_index().drop(columns=['index'])

X_train, X_test, y_train, y_test = train_test_split(df_upsampled.loc[:, df_upsampled.columns != 'Survived'],
                                                    df_upsampled['Survived'], test_size=0.3, random_state=2)

f1 = plt.figure()
ax1 = f1.add_subplot(111)

models = [LogisticRegression(max_iter=1000), DecisionTreeClassifier(max_depth=10),
          RandomForestClassifier(max_depth=12, random_state=10)]
for model in models:
    create_model(model, X_train, y_train, X_test, y_test, ax1)

ax1.plot([0, 1], [0, 1], 'r--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
ax1.legend()
plt.show()


plt.show()
