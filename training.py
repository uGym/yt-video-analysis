import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re


def iso8601_duration_to_seconds(duration):
    duration_regex = re.compile(r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?')
    matches = duration_regex.match(duration)
    if not matches:
        return 0

    hours = int(matches.group('hours')) if matches.group('hours') else 0
    minutes = int(matches.group('minutes')) if matches.group('minutes') else 0
    seconds = int(matches.group('seconds')) if matches.group('seconds') else 0

    return hours * 3600 + minutes * 60 + seconds

df = pd.read_csv('video_data.csv')
df['duration_seconds'] = df['duration'].apply(iso8601_duration_to_seconds)

X = df[['title_length', 'duration_seconds']]
y = df['view_count']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

rf_model = RandomForestClassifier(criterion='gini')

rf_model.fit(X_train, y_train)

y_val_pred = rf_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

y_test_pred = rf_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')