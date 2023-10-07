import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model

from util import video_score, scaled_score

EPOCHS = 100
BATCH_SIZE = 64
HIDDEN_LAYER1_NEURONS = 64
HIDDEN_LAYER2_NEURONS = 32
ACTIVATION_FUNC1 = 'relu'
ACTIVATION_FUNC2 = 'relu'
OPTIMIZER = 'adam'
LOSS_FUNC = 'sparse_categorical_crossentropy'

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
        'special_characters_count', 'emoji_count', 'at_tags_count', 'description_length',
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

model = Sequential()
model.add(Dense(HIDDEN_LAYER1_NEURONS, input_dim=X_train_scaled.shape[1], activation=ACTIVATION_FUNC1))
model.add(Dense(HIDDEN_LAYER2_NEURONS, activation=ACTIVATION_FUNC2))
model.add(Dense(len(le.classes_), activation='softmax'))

model.compile(loss=LOSS_FUNC, optimizer=OPTIMIZER, metrics=['accuracy'])

model.fit(X_train_scaled, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val_scaled, y_val))

val_loss, val_accuracy = model.evaluate(X_val_scaled, y_val)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)

print(f'Validation Accuracy: {val_accuracy}')
print(f'Test Accuracy: {test_accuracy}')

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
