import csv
import os
import re
import time
from datetime import datetime, timezone

import cv2
import face_recognition
import numpy as np
import requests

from util import crop_to_16_9, find_dominant_color, entropy, iso8601_duration_to_seconds

channel_data_dict = {}
api_key = "YOUTUBE_API_KEY"

def save_to_csv(data, filename="data/video_data.csv"):
    mode = 'a' if os.path.isfile(filename) else 'w'
    with open(filename, mode, newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if mode == 'w':
            writer.writerow(["video_id", "video_title", "title_length", "uppercase_count", "lowercase_count",
                             "special_characters_count", "emoji_count", "at_tags_count", "dominant_color_R",
                             "dominant_color_G", "dominant_color_B", "num_edges", "img_entropy", "description_length",
                             "hashtags_count", "urls_count", "tag_count", "duration", "view_count", "like_count",
                             "comment_count", "subscriber_count", "seconds_since_upload", "num_faces"])
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

def fetch_channel_data(channel_id, api_key):
    if channel_id not in channel_data_dict:
        channel_data = requests.get(
            "https://www.googleapis.com/youtube/v3/channels",
            params={"id": channel_id, "key": api_key, "part": "statistics"}
        ).json()['items'][0]
        channel_data_dict[channel_id] = channel_data

    return channel_data_dict[channel_id]

def video_details(video_ids, api_key):
    items = requests.get(
        "https://www.googleapis.com/youtube/v3/videos",
        params={"id": ','.join(video_ids), "key": api_key, "part": "snippet,statistics,contentDetails"}
    ).json()['items']

    details = {}
    for item in items:
        channel_id = item['snippet']['channelId']
        channel_data = fetch_channel_data(channel_id, api_key)
        published_at = item['snippet']['publishedAt']
        published_at_datetime = datetime.fromisoformat(published_at[:-1]).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        seconds_since_upload = (now - published_at_datetime).total_seconds()

        data = {
            "title": item['snippet']['title'],
            "description": item['snippet']['description'].replace('\n', ' '),
            "tag_count": len(item['snippet'].get('tags', [])),
            "duration": item['contentDetails']['duration'],
            "view_count": int(item['statistics']['viewCount']),
            "like_count": int(item['statistics'].get('likeCount', 0)),
            "comment_count": int(item['statistics'].get('commentCount', 0)),
            "subscriber_count": int(channel_data['statistics'].get('subscriberCount', 0)),
            "seconds_since_upload": int(seconds_since_upload)
        }
        details[item['id']] = data

    return details


def thumbnail_url(video_id, api_key):
    url = "https://www.googleapis.com/youtube/v3/videos?id={}&key={}&part=snippet".format(video_id, api_key)
    return requests.get(url).json()['items'][0]['snippet']['thumbnails']['high']['url']


def fetch_thumbnail(url):
    img = cv2.imdecode(np.frombuffer(requests.get(url).content, np.uint8), -1)
    return crop_to_16_9(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def thumbnail_details(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return find_dominant_color(img), np.count_nonzero(edges), entropy(cv2.calcHist([gray], [0], None, [256], [0, 256])), len(face_recognition.face_locations(img))


def fetch_video_ids(api_key, category_id, max_results, country=None, language=None):
    video_ids = []
    params = {
        'part': 'id',
        'videoCategoryId': category_id,
        'type': 'video',
        'key': api_key,
        'maxResults': 100,
        'videoDuration': 'medium',
        'fields': 'items/id,nextPageToken'
    }
    if country:
        params['regionCode'] = country
    if language:
        params['relevanceLanguage'] = language

    fetched = 0
    while max_results > 0:
        response = requests.get("https://www.googleapis.com/youtube/v3/search", params=params).json()
        if 'items' in response:
            for item in response['items']:
                video_ids.append(item['id']['videoId'])
                max_results -= 1
                fetched += 1
                if max_results <= 0:
                    break
            if 'nextPageToken' in response:
                params['pageToken'] = response['nextPageToken']
            else:
                break
        else:
            break
        time.sleep(1)
    return video_ids


def fetch_data(max_results):
    video_ids = fetch_video_ids(api_key, 37, max_results)
    counter = 1

    batch_size = 50

    for i in range(0, len(video_ids), batch_size):
        batch_ids = video_ids[i:i + batch_size]
        details = video_details(batch_ids, api_key)

        for video_id, video_data in details.items():
            image = fetch_thumbnail(thumbnail_url(video_id, api_key))
            thumb_data = thumbnail_details(image)
            title_desc_data = title_description_data(video_data["title"], video_data["description"])

            dominant_color_R, dominant_color_G, dominant_color_B = thumb_data[0]

            data = [
                video_id,
                video_data["title"],
                title_desc_data["title_length"],
                title_desc_data["uppercase_count"],
                title_desc_data["lowercase_count"],
                title_desc_data["special_characters_count"],
                title_desc_data["emoji_count"],
                title_desc_data["at_tags_count"],
                dominant_color_R,
                dominant_color_G,
                dominant_color_B,
                thumb_data[1],
                thumb_data[2],
                title_desc_data["description_length"],
                title_desc_data["hashtags_count"],
                title_desc_data["urls_count"],
                video_data["tag_count"],
                iso8601_duration_to_seconds(video_data["duration"]),
                video_data["view_count"],
                video_data["like_count"],
                video_data["comment_count"],
                video_data["subscriber_count"],
                video_data["seconds_since_upload"],
                thumb_data[3]
            ]

            save_to_csv(data)

            print(f"Processed video: {video_id} ID={counter}: {video_data['title']}")
            counter += 1