import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from scipy import stats
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, plot_confusion_matrix, precision_score, recall_score, f1_score, \
    roc_auc_score, roc_curve
import time
# set pycharm terminal wider than default
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 14)

PATH = '/home/tav/Desktop/Untitled Folder/data/'
dataset = pd.read_csv(f'{PATH}train.csv')
test_data = pd.read_csv(f'{PATH}test.csv')

print(train_data.info())
print(train_data.describe())
remove_cols = ['Ticket', 'Cabin', 'Cabin']

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

# Downsample majority class
df_majority_downsampled = resample(df_majority, replace=False, n_samples=len(dataset[dataset['Survived'] == 1]),
                                   random_state=123)
df_downsampled = pd.concat([df_majority_downsampled, df_minority])
df_downsampled = df_downsampled.sample(frac=1, random_state=30).reset_index().drop(columns=['index'])

# Upsample minority class
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(dataset[dataset['Survived'] == 0]),
                                 random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled = df_upsampled.sample(frac=1, random_state=30).reset_index().drop(columns=['index'])

# no na values now
# Subset data into train and test
# X_train, X_test, y_train, y_test = train_test_split(df_downsampled.loc[:, df_downsampled.columns != 'Survived'],
#                                                     df_downsampled['Survived'], test_size=0.3, random_state=2)
X_train, X_test, y_train, y_test = train_test_split(df_upsampled.loc[:, df_upsampled.columns != 'Survived'],
                                                    df_upsampled['Survived'], test_size=0.3, random_state=2)


# Plot Results
fig = plt.figure(figsize=(14, 4))
(ax1, ax2, ax3) = fig.subplots(1, 3)
fig.suptitle('ML Model Confusion Matrices', size=16)

# Logistic Regression
lr = LogisticRegression(n_jobs=-1, max_iter=1000)
time_start_lr = time.perf_counter()
lr_model = lr.fit(X_train, y_train)
time_elapsed_lr = (time.perf_counter() - time_start_lr)
# Model Values
y_pred_lr = lr_model.predict(X_test)
prob_lr = lr.predict_proba(X_test)[:, 1]
lr_accuracy = accuracy_score(y_test, y_pred_lr) * 100
lr_precision = precision_score(y_test, y_pred_lr) * 100
lr_recall = recall_score(y_test, y_pred_lr) * 100
plot_confusion_matrix(lr_model, X_test, y_test, ax=ax1, normalize='pred')
ax1.title.set_text('Logistic Regression Confusion Matrix')

# Decision Tree
dt = DecisionTreeClassifier(max_depth=10, random_state=1)
time_start_dt = time.perf_counter()
dt_model = dt.fit(X_train, y_train)
time_elapsed_dt = (time.perf_counter() - time_start_dt)
# Model Values
y_pred_dt = dt_model.predict(X_test)
prob_dt = dt.predict_proba(X_test)[:, 1]
dt_accuracy = accuracy_score(y_test, y_pred_dt) * 100
dt_precision = precision_score(y_test, y_pred_dt) * 100
dt_recall = recall_score(y_test, y_pred_dt) * 100
plot_confusion_matrix(dt_model, X_test, y_test, ax=ax2, normalize='pred')
ax2.title.set_text('Decision Tree Confusion Matrix')

# Random Forest
rf = RandomForestClassifier(max_depth=12, random_state=10)
time_start_rf = time.perf_counter()
rf_model = rf.fit(X_train, y_train)
time_elapsed_rf = (time.perf_counter() - time_start_rf)
# Model Values
y_pred_rf = rf_model.predict(X_test)
prob_rf = rf.predict_proba(X_test)[:, 1]
rf_accuracy = accuracy_score(y_test, y_pred_rf) * 100
rf_precision = precision_score(y_test, y_pred_rf) * 100
rf_recall = recall_score(y_test, y_pred_rf) * 100
plot_confusion_matrix(rf_model, X_test, y_test, ax=ax3, normalize='pred')
ax3.title.set_text('Random Forest Confusion Matrix')
plt.show()

# ROC Curves
auc_lr = roc_auc_score(y_test, prob_lr)
auc_dt = roc_auc_score(y_test, prob_dt)
auc_rf = roc_auc_score(y_test, prob_rf)

lr_fpr, lr_tpr, _ = roc_curve(y_test, prob_lr)
dt_fpr, dt_tpr, _ = roc_curve(y_test, prob_dt)
rf_fpr, rf_tpr, _ = roc_curve(y_test, prob_rf)

plt.plot([0, 1], [0, 1], 'r--')
plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic Regression')
plt.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
plt.plot(rf_fpr, rf_tpr, marker='.', label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# Logistic Regression Metrics
print("Logistic Regression Metrics")
print("LR Computation Time:%5.4f seconds" % time_elapsed_lr)
print("LR Accuracy: %.4f" % lr_accuracy)
print("LR Precision: %.4f" % lr_precision)
print("LR Recall: %.4f" % lr_recall)
print("LR AUC: %.4f" % auc_lr, "\n")

# Decision Tree Metrics
print("Decision Tree Metrics")
print("DT Computation Time:%5.4f seconds" % time_elapsed_dt)
print("DT Accuracy: %.4f" % dt_accuracy)
print("DT Precision: %.4f" % dt_precision)
print("DT Recall: %.4f" % dt_recall)
print("DT AUC: %.4f" % auc_dt, "\n")

# Random Forest Metrics
print("Decision Tree Random Forest Metrics")
print("DTRF Computation Time:%5.4f seconds" % time_elapsed_rf)
print("DTRF Accuracy: %.4f" % rf_accuracy)
print("DTRF Precision: %.4f" % rf_precision)
print("DTRF Recall: %.4f" % rf_recall)
print("DTRF AUC: %.4f" % auc_rf)
