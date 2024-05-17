import re
import requests

from bs4 import BeautifulSoup
from youtube_transcript_api import YouTubeTranscriptApi


def get_transcript_from_url(video_url):
    video_id = extract_youtube_code(video_url)
    video_title = extract_video_title(video_url)
    video_channel = extract_youtube_channel(video_url)
    trancript = YouTubeTranscriptApi.get_transcript(video_id)
    return merge_transcripts(trancript, video_title, video_channel)

def extract_youtube_code(url):
    return re.findall("v=(.*)_channel?", url)[0]

def extract_youtube_channel(url):
    channel = re.findall("channel=(.*)", url)
    if len(channel) > 0:
        return channel[0]
    return []

def extract_video_title(url):
    r = requests.get(url)
    soup = BeautifulSoup(r.text)

    link = soup.find_all(name="title")[0]
    title = str(link)
    title = title.replace("<title>","")
    title = title.replace("</title>","")
    return title

def merge_transcripts(trancript, video_title, video_channel, max_duration=25):
    all_merged_transcripts = []
    merged_transcript = {"text": "", "duration": 0, "title": video_title, "channel": video_channel}
    for wording in trancript:
        merged_transcript["text"] += wording["text"]
        merged_transcript["duration"] += wording["duration"]
        if merged_transcript["duration"] >= max_duration:
            all_merged_transcripts.append(merged_transcript)
            merged_transcript = {"text": "", "duration": 0, "title": video_title, "channel": video_channel}
    return all_merged_transcripts