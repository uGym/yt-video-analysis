import requests
import cv2
import numpy as np
import csv
import os
import re

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from scipy.stats import entropy

api_key = "AIzaSyAlQrNzNL4VUyw0jmlRjsX_C_45r7KIMh0"
video_id = "lJrHLnhJl-M"

def fetch_thumbnail(url):
    response = requests.get(url)
    image = cv2.imdecode(np.frombuffer(response.content, np.uint8), -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return crop_to_16_9(image)


def thumbnail_details(image):
    dominant_color = find_dominant_color(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    num_edges = np.count_nonzero(edges)
    hist_gray = cv2.calcHist([gray], [0], None, [256], [0, 256])
    img_entropy = entropy(hist_gray)
    return dominant_color, num_edges, img_entropy

def crop_to_16_9(image):
    height, width, _ = image.shape
    desired_width = int((16/9) * height)
    desired_height = int((9/16) * width)

    if width > desired_width:
        x1 = (width - desired_width) // 2
        x2 = x1 + desired_width
        cropped_image = image[:, x1:x2]
    elif height > desired_height:
        y1 = (height - desired_height) // 2
        y2 = y1 + desired_height
        cropped_image = image[y1:y2, :]

    else:
        cropped_image = image
    return cropped_image

def find_dominant_color(image, sample_size=10000, clusters=5):
    idx = np.random.choice(image.shape[0] * image.shape[1], sample_size, replace=False)
    samples = image.reshape((-1, 3))[idx]
    kmeans = KMeans(n_clusters=clusters, n_init=10).fit(samples)
    dominant_colors = kmeans.cluster_centers_
    labels, counts = np.unique(kmeans.labels_, return_counts=True)
    dominant_cluster = labels[np.argmax(counts)]
    dominant_color = dominant_colors[dominant_cluster].astype(np.uint8)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    dominant_color_img = np.zeros((100, 100, 3), dtype=np.uint8)
    dominant_color_img[:, :] = dominant_color
    plt.imshow(dominant_color_img)
    plt.title('Dominant Color')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    return dominant_color

def fetch_video_ids(api_key, query, max_results):
    base_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        'part': 'id',
        'q': query,
        'type': 'video',
        'key': api_key,
        'maxResults': max_results
    }

    response = requests.get(base_url, params=params)
    results = response.json()

    video_ids = [item['id']['videoId'] for item in results['items']]
    return video_ids

def title_description_data(title, description):
    title_length = len(title)
    uppercase_count = sum(1 for char in title if char.isupper())
    lowercase_count = sum(1 for char in title if char.islower())
    special_characters_count = len(re.findall(r'[^\w\s]', title))
    emoji_count = len(re.findall(r'[\U00010000-\U0010ffff]', title))
    at_tags_count = title.count('@')

    description_length = len(description)
    hashtags_count = description.count('#')
    urls_count = len(
        re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', description))

    return {
        "title_length": title_length,
        "uppercase_count": uppercase_count,
        "lowercase_count": lowercase_count,
        "special_characters_count": special_characters_count,
        "emoji_count": emoji_count,
        "at_tags_count": at_tags_count,
        "description_length": description_length,
        "hashtags_count": hashtags_count,
        "urls_count": urls_count
    }


def video_details(video_id, api_key):
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "id": video_id,
        "key": api_key,
        "part": "snippet,statistics,contentDetails"
    }
    response = requests.get(base_url, params=params)
    data = response.json()

    item = data['items'][0]

    title = item['snippet']['title']
    description = item['snippet']['description'].replace('\n', ' ')
    tags = item['snippet'].get('tags', [])
    duration = item['contentDetails']['duration']
    view_count = int(item['statistics']['viewCount'])
    like_count = int(item['statistics'].get('likeCount', 0))

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "duration": duration,
        "view_count": view_count,
        "like_count": like_count
    }


def save_to_csv(data, filename="video_data.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            header = ["video_id",
                      "title_length", "uppercase_count", "lowercase_count", "special_characters_count", "emoji_count",
                      "at_tags_count",
                      "dominant_color_r", "dominant_color_g", "dominant_color_b",
                      "num_edges", "img_entropy",
                      "description_length", "hashtags_count", "urls_count",
                      "tags", "duration", "view_count", "like_count"]
            writer.writerow(header)
        writer.writerow(data)


def thumbnail_url(video_id, api_key):
    base_url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "id": video_id,
        "key": api_key,
        "part": "snippet"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    thumbnail_url = data['items'][0]['snippet']['thumbnails']['high']['url']
    return thumbnail_url


video_ids = fetch_video_ids(api_key, "gaming", 200)

for id in video_ids:
    image = fetch_thumbnail(thumbnail_url(id, api_key))
    dominant_color, num_edges, img_entropy = thumbnail_details(image)
    video_details_data = video_details(id, api_key)
    title_desc_data = title_description_data(video_details_data["title"], video_details_data["description"])
    data = [id,
            title_desc_data["title_length"],
            title_desc_data["uppercase_count"],
            title_desc_data["lowercase_count"],
            title_desc_data["special_characters_count"],
            title_desc_data["emoji_count"],
            title_desc_data["at_tags_count"],
            dominant_color[0], dominant_color[1], dominant_color[2],
            num_edges, img_entropy,
            title_desc_data["description_length"],
            title_desc_data["hashtags_count"],
            title_desc_data["urls_count"],
            ",".join(video_details_data["tags"]),
            video_details_data["duration"],
            video_details_data["view_count"],
            video_details_data["like_count"]]

    save_to_csv(data)

