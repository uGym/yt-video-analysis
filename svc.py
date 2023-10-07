import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from util import video_score, scaled_score

df = pd.read_csv('video_data.csv')
df = df.drop_duplicates(subset='video_id')
print("Number of rows:", df.shape[0])

df['score'] = df.apply(lambda row: video_score(row['view_count'], row['subscriber_count'], row['seconds_since_upload']), axis=1)
highest_score = df['score'].max()
df['scaled_score'] = df['score'].apply(lambda x: scaled_score(x, highest_score))

lower_quantile = df['scaled_score'].quantile(0.33)
upper_quantile = df['scaled_score'].quantile(0.66)
df['category'] = pd.cut(df['scaled_score'], bins=[0, lower_quantile, upper_quantile, float('inf')], labels=['low', 'medium', 'high'])

X = df[['title_length', 'duration', 'uppercase_count', 'lowercase_count',
        'special_characters_count', 'emoji_count', 'at_tags_count',
        'hashtags_count', 'urls_count', 'num_edges', 'num_faces', 'img_entropy', 'dominant_color_R',
        'dominant_color_G', 'dominant_color_B']]
y = df['category']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train_scaled, y_train)

y_val_pred = svm_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

y_test_pred = svm_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(10, 8))
importance = svm_model.coef_[0]
plt.barh(range(len(importance)), importance)
plt.yticks(range(len(importance)), X.columns)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.show()