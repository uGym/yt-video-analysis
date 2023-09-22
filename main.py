import requests
import cv2
import numpy as np
import face_recognition
import re
import csv
import os
from sklearn.cluster import KMeans
from scipy.stats import entropy as scipy_entropy


api_key = "YOUTUBE_API_KEY"


def save_to_csv(data, filename="video_data.csv"):
    mode = 'a' if os.path.isfile(filename) else 'w'
    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(["video_id", "video_title", "title_length", "uppercase_count", "lowercase_count",
                             "special_characters_count", "emoji_count", "at_tags_count", "dominant_color_hex",
                             "num_edges", "img_entropy", "description_length", "hashtags_count", "urls_count",
                             "tags", "duration", "view_count", "like_count", "comment_count", "num_faces"])
        writer.writerow(data)


def title_description_data(title, description):
    data = {
        "title_length": len(title),
        "uppercase_count": sum(1 for c in title if c.isupper()),
        "lowercase_count": sum(1 for c in title if c.islower()),
        "special_characters_count": len(re.findall(r'[^\w\s]', title)),
        "emoji_count": len(re.findall(r'[\U00010000-\U0010ffff]', title)),
        "at_tags_count": title.count('@'),
        "description_length": len(description),
        "hashtags_count": description.count('#'),
        "urls_count": len(re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description))
    }
    return data


def video_details(video_id, api_key):
    item = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"id": video_id, "key": api_key, "part": "snippet,statistics,contentDetails"}
    ).json()['items'][0]

    data = {
        "title": item['snippet']['title'],
        "description": item['snippet']['description'].replace('\n', ' '),
        "tags": item['snippet'].get('tags', []),
        "duration": item['contentDetails']['duration'],
        "view_count": int(item['statistics']['viewCount']),
        "like_count": int(item['statistics'].get('likeCount', 0)),
        "comment_count": int(item['statistics'].get('commentCount', 0))
    }

    return data


def thumbnail_url(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/videos?id={}&key={}&part=snippet".format(video_id, api_key)
    return requests.get(url).json()['items'][0]['snippet']['thumbnails']['high']['url']


def fetch_thumbnail(url):
    img = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), -1)
    return crop_to_16_9(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def entropy(hist):
    hist = hist / np.sum(hist)
    return scipy_entropy(hist)


def thumbnail_details(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return find_dominant_color(img), np.count_nonzero(edges), entropy(cv2.calcHist([gray], [0], None, [256], [0, 256])), len(face_recognition.face_locations(img))


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
    return "#{:02x}{:02x}{:02x}".format(*kmeans.cluster_centers_[labels[np.argmax(counts)]].astype(np.uint8))


def fetch_video_ids(api_key, query, max_results):
    return [item['id']['videoId'] for item in requests.get("https://www.googleapis.com/youtube/v3/search", params={'part': 'id','q': query,'type': 'video','key': api_key,'maxResults': max_results}).json()['items']]


video_ids = fetch_video_ids(api_key, "gaming", 200)

for id in video_ids:
    image = fetch_thumbnail(thumbnail_url(id, api_key))
    thumb_data = thumbnail_details(image)
    video_data = video_details(id, api_key)
    title_desc_data = title_description_data(video_data["title"], video_data["description"])

    data = [
        id,
        video_data["title"],
        title_desc_data["title_length"],
        title_desc_data["uppercase_count"],
        title_desc_data["lowercase_count"],
        title_desc_data["special_characters_count"],
        title_desc_data["emoji_count"],
        title_desc_data["at_tags_count"],
        thumb_data[0],
        thumb_data[1],
        thumb_data[2],
        title_desc_data["description_length"],
        title_desc_data["hashtags_count"],
        title_desc_data["urls_count"],
        ",".join(video_data["tags"]),
        video_data["duration"],
        video_data["view_count"],
        video_data["like_count"],
        video_data["comment_count"],
        thumb_data[3]
    ]

    save_to_csv(data)