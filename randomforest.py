import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

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

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestClassifier(criterion='gini', max_depth=10, min_samples_leaf=4, min_samples_split=5, n_estimators=200)

rf_model.fit(X_train_scaled, y_train)

importances = rf_model.feature_importances_
feature_names = X.columns.tolist()
sorted_indices = np.argsort(importances)[::-1]

y_val_pred = rf_model.predict(X_val_scaled)
val_accuracy = accuracy_score(y_val, y_val_pred)

y_test_pred = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

single_tree = rf_model.estimators_[0]

plt.figure(figsize=(20,10))
plot_tree(single_tree, filled=True, feature_names=X.columns.tolist(), class_names=['low', 'medium', 'high'], rounded=True)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(30, 10))

plot_tree(single_tree, filled=True, feature_names=X.columns.tolist(), class_names=['low', 'medium', 'high'], rounded=True, ax=axes[0])
axes[0].set_title('Decision Tree')

axes[1].barh(range(len(sorted_indices)), importances[sorted_indices], align='center')
axes[1].set_yticks(range(len(sorted_indices)))
axes[1].set_yticklabels([feature_names[i] for i in sorted_indices])
axes[1].invert_yaxis()
axes[1].set_xlabel('Feature Importance')
axes[1].set_title('Feature Importances')

plt.tight_layout()
plt.show()