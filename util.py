import math
import re

import numpy as np
from scipy.stats import entropy as scipy_entropy
from sklearn.cluster import KMeans


def video_score(view_count, subscriber_count, seconds_since_upload):
    if subscriber_count == 0 or seconds_since_upload == 0:
        return 0
    days_since_upload = seconds_since_upload / 86400
    time_decay = 1 + math.log1p(days_since_upload) * 1.035
    return (math.pow(view_count, 0.71) / math.pow(subscriber_count, 0.98) * 1.0) * 1000 / time_decay

def scaled_score(score, highest_score):
    scaled_score = (score / highest_score) * 1000
    return scaled_score

def custom_score(alpha, beta, gamma, delta, view_count, subscriber_count, seconds_since_upload):
    if subscriber_count == 0 or seconds_since_upload == 0:
        return 0
    days_since_upload = seconds_since_upload / 86400
    time_decay = 1 + math.log1p(days_since_upload) * beta
    return (math.pow(view_count, gamma) / math.pow(subscriber_count, alpha) * delta) * 1000 / time_decay


def iso8601_duration_to_seconds(duration):
    duration_regex = re.compile(r'PT((?P<hours>\d+)H)?((?P<minutes>\d+)M)?((?P<seconds>\d+)S)?')
    matches = duration_regex.match(duration)
    if not matches:
        return 0

    hours = int(matches.group('hours')) if matches.group('hours') else 0
    minutes = int(matches.group('minutes')) if matches.group('minutes') else 0
    seconds = int(matches.group('seconds')) if matches.group('seconds') else 0

    return hours * 3600 + minutes * 60 + seconds


def entropy(hist):
    hist = hist / np.sum(hist)
    ent = scipy_entropy(hist)
    return int(round(ent.item()))

def crop_to_16_9(image):
    h, w, _ = image.shape
    dw, dh = int(16/9 * h), int(9/16 * w)
    if w > dw:
        x = (w - dw) // 2
        return image[:, x:x+dw]
    elif h > dh:
        y = (h - dh) // 2
        return image[y:y+dh, :]
    return image


def find_dominant_color(img, sample_size=10000, clusters=5):
    samples = img.reshape((-1, 3))[np.random.choice(img.shape[0] * img.shape[1], sample_size, replace=False)]
    kmeans = KMeans(n_clusters=clusters, n_init=10).fit(samples)
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_color = kmeans.cluster_centers_[labels[np.argmax(counts)]].astype(np.uint8)
    return dominant_color[0], dominant_color[1], dominant_color[2]