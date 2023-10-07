import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import xgboost as xgb
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

model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_))
model.fit(X_train_scaled, y_train)

y_val_pred = model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

y_test_pred = model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

mapper = {'f{0}'.format(i): v for i, v in enumerate(X.columns)}
mapped = {mapper[k]: v for k, v in model.get_booster().get_fscore().items()}

plt.figure(figsize=(20, 10))
ax = xgb.plot_importance(mapped)
for item in ax.get_xticklabels():
    item.set_rotation(45)
plt.tight_layout()
plt.show()
