import math

def video_score(view_count, subscriber_count, seconds_since_upload):
    if subscriber_count == 0 or seconds_since_upload == 0:
        return 0
    days_since_upload = seconds_since_upload / 86400
    time_decay = 1 + math.log1p(days_since_upload)
    return (view_count / math.sqrt(subscriber_count)) * 1000 / time_decay

def scaled_score(score, highest_score):
    scaled_score = (score / highest_score) * 1000
    return scaled_score
