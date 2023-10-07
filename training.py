import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import re

def video_score(view_count, subscriber_count, seconds_since_upload):
    if subscriber_count == 0 or seconds_since_upload == 0:
        return 0
    days_since_upload = seconds_since_upload / 86400
    time_decay = 1 + math.log1p(days_since_upload)
    return (view_count / math.sqrt(subscriber_count)) * 1000 / time_decay

def iso8601_duration_to_seconds(duration):
    duration_regex = re.compile(r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?')
    matches = duration_regex.match(duration)
    if not matches:
        return 0

    hours = int(matches.group('hours')) if matches.group('hours') else 0
    minutes = int(matches.group('minutes')) if matches.group('minutes') else 0
    seconds = int(matches.group('seconds')) if matches.group('seconds') else 0

    return hours * 3600 + minutes * 60 + seconds

def scaled_score(score, highest_score):
    scaled_score = (score / highest_score) * 1000
    return scaled_score

df = pd.read_csv('video_data.csv')
df['duration_seconds'] = df['duration'].apply(iso8601_duration_to_seconds)

df['score'] = df.apply(lambda row: video_score(row['view_count'], row['subscriber_count'], row['seconds_since_upload']), axis=1)
highest_score = df['score'].max()

df['scaled_score'] = df['score'].apply(lambda x: scaled_score(x, highest_score))
lower_quantile = df['scaled_score'].quantile(0.33)
upper_quantile = df['scaled_score'].quantile(0.66)
df['category'] = pd.cut(df['scaled_score'], bins=[0, lower_quantile, upper_quantile, float('inf')], labels=['low', 'medium', 'high'])

X = df[['title_length', 'duration_seconds']]
y = df['category']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

rf_model = RandomForestClassifier(criterion='gini')

rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

plt.figure(figsize=(40, 20))
plot_tree(rf_model.estimators_[0], filled=True, feature_names=X.columns.tolist(), class_names=list(map(str, y.unique())), rounded=True)
plt.savefig('tree_high_res.png', dpi=300)
plt.show()

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')