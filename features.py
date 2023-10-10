import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from util import scaled_score, video_score

LOWER_QUANTILE_RATIO = 0.25
UPPER_QUANTILE_RATIO = 0.75
TRAIN_SIZE_RATIO = 0.8
VAL_TEST_SPLIT_RATIO = 0.5
RANDOM_STATE = 42


def preprocess_data(file_path, one_hot_encode=False):
    df = pd.read_csv(file_path)
    df = df.drop_duplicates(subset='video_id')

    df['score'] = df.apply(lambda row: video_score(row['view_count'], row['subscriber_count'], row['seconds_since_upload']), axis=1)
    highest_score = df['score'].max()
    df['scaled_score'] = df['score'].apply(lambda x: scaled_score(x, highest_score))

    lower_quantile = df['scaled_score'].quantile(LOWER_QUANTILE_RATIO)
    upper_quantile = df['scaled_score'].quantile(UPPER_QUANTILE_RATIO)
    df['category'] = pd.cut(df['scaled_score'], bins=[0, lower_quantile, upper_quantile, float('inf')], labels=['low', 'medium', 'high'])

    df['upper_case_ratio'] = df['uppercase_count'] / df['title_length']
    df['lower_case_ratio'] = df['lowercase_count'] / df['title_length']

    X = df[['title_length', 'duration', 'lower_case_ratio',
            'upper_case_ratio', 'special_characters_count',
            'emoji_count', 'at_tags_count', 'hashtags_count',
            'num_edges', 'num_faces', 'img_entropy',
            'dominant_color_R', 'dominant_color_G', 'dominant_color_B']]

    y = df['category']

    y.dropna(inplace=True)
    X = X.loc[y.index]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-TRAIN_SIZE_RATIO, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=VAL_TEST_SPLIT_RATIO, random_state=RANDOM_STATE)

    if one_hot_encode:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = onehot_encoder.fit_transform(np.array(y_train).reshape(-1, 1))
        y_val = onehot_encoder.transform(np.array(y_val).reshape(-1, 1))
        y_test = onehot_encoder.transform(np.array(y_test).reshape(-1, 1))
        num_classes = y_train.shape[1]
    else:
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_val = le.transform(y_val)
        y_test = le.transform(y_test)
        num_classes = len(le.classes_)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    feature_names = X.columns.tolist()

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, feature_names, num_classes
