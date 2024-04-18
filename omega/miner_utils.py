import os
import time
from typing import List, Tuple

import bittensor as bt

from omega.protocol import VideoMetadata, Videos
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES, VALIDATOR_TIMEOUT
from omega import video_utils

import asyncio
import traceback
import sqlite3
import random
import requests
import json
import re
import shutil
import numpy as np

import torch
import torch.nn.functional as F

# toggle to optimize queries or not
OPTIMIZE_QUERIES = False
# toggle to turn on description similarity checks
DO_DESCRIPTION_SIMILARITY = False
# cosign_similarity threshold. We want similarities at or above this number.
SIMILARITY_THRESHOLD = 0.25
# threshold for validator timeout. if we're within this many seconds, abort our loops and send back what we have 
TIMEOUT_THRESHOLD = 10

# Read proxy URLs from our file so we don't have to update our code every time =)
PROXY_URLS = []
PROXY_FILENAME = "proxies.txt"
with open(PROXY_FILENAME, 'r') as file:
    for line in file:
        proxy_url = line.strip()
        if proxy_url != "":
            PROXY_URLS.append(proxy_url)
            
""" QUERY AUGMENTATION COLLECTION LOGIC """     
db_filename_queries = 'queries.sqlite'
conn = sqlite3.connect(db_filename_queries)
cursor = conn.cursor()
create_table_query = '''
CREATE TABLE IF NOT EXISTS queries (
    query CHAR(50), 
    augment CHAR(100) PRIMARY KEY,
    count INTEGER NOT NULL DEFAULT 0
)
'''
cursor.execute(create_table_query)
conn.commit()
conn.close()

# check if an augment exists with more than max_count uses
def check_query_augment(augment, max_count=3):
    conn = sqlite3.connect(db_filename_queries)
    cursor = conn.cursor()
    query = 'SELECT EXISTS(SELECT 1 FROM queries WHERE augment=? AND count >= ? LIMIT 1)'
    cursor.execute(query, (augment,max_count))
    # Fetch the result
    exists = cursor.fetchone()[0]
    conn.close()
    # Return True if exists is 1, otherwise False
    return exists == 1
    
# find an eligible query augment with less than max_count uses
def find_query_augment(search_query, max_count=3):
    conn = sqlite3.connect(db_filename_queries)
    cursor = conn.cursor()
    query = 'SELECT augment FROM queries WHERE query=? AND count < ? ORDER BY count ASC LIMIT 1'
    cursor.execute(query, (search_query,max_count))
    # Fetch the result
    augment = cursor.fetchone()
    conn.close()
    # Check if result is not None, then return the augment value, otherwise return False
    return augment[0] if augment else False
    
# find query augments based on search_query. return limit rows
def find_query_augments(search_query, search_augment, limit=5):
    conn = sqlite3.connect(db_filename_queries)
    cursor = conn.cursor()
    query = f"SELECT augment FROM queries WHERE query=? AND augment != ? ORDER BY RANDOM() LIMIT ?"
    cursor.execute(query, (search_query, search_augment, limit))
    # Fetch the result and extract the first element from each tuple
    augments = [item[0] for item in cursor.fetchall()]
    conn.close()
    return augments
    
def update_query_augment(augment):
    conn = sqlite3.connect(db_filename_queries)
    cursor = conn.cursor()
    # SQL query to update augmented query. Add one to count.
    query = 'UPDATE queries SET count = count+1 WHERE augment = ?'
    cursor.execute(query, (augment,))
    conn.commit()
    conn.close()
	
def insert_query_augments(queriesArr):
    conn = sqlite3.connect(db_filename_queries)
    cursor = conn.cursor()
    # SQL query to insert or ignore the new augmented queries. Bulk upload
    insert_query = 'INSERT OR IGNORE INTO queries (query, augment) VALUES (?, ?)'
    cursor.executemany(insert_query, queriesArr)
    conn.commit()
    conn.close()

""" YOUTUBE VIDEO ID COLLECTION LOGIC """
db_filename = 'youtube.sqlite'
conn = sqlite3.connect(db_filename)
cursor = conn.cursor()
create_table_query = '''
CREATE TABLE IF NOT EXISTS youtube_videos (
    youtube_id CHAR(30) PRIMARY KEY
)
'''
cursor.execute(create_table_query)
conn.commit()
conn.close()

def check_youtube_id(youtube_id):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
	# SQL query to check for the existence of the youtube_id
    query = 'SELECT EXISTS(SELECT 1 FROM youtube_videos WHERE youtube_id=? LIMIT 1)'
    cursor.execute(query, (youtube_id,))
    # Fetch the result
    exists = cursor.fetchone()[0]
    conn.close()
    # Return True if exists is 1, otherwise False
    return exists == 1
	
def insert_youtube_id(youtube_id):
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    # SQL query to insert or ignore the new youtube_id
    insert_query = 'INSERT OR IGNORE INTO youtube_videos (youtube_id) VALUES (?)'
    cursor.execute(insert_query, (youtube_id,))
    conn.commit()
    conn.close()
    
""" YOUTUBE VIDEO IDs longer than 5 minutes """
db_filename_yt5 = 'youtube5m.sqlite'
conn = sqlite3.connect(db_filename_yt5)
cursor = conn.cursor()
create_table_query = '''
CREATE TABLE IF NOT EXISTS youtube_videos (
    youtube_id CHAR(30) PRIMARY KEY
)
'''
cursor.execute(create_table_query)
conn.commit()
conn.close()

def check_youtube5m_id(youtube_id):
    conn = sqlite3.connect(db_filename_yt5)
    cursor = conn.cursor()
	# SQL query to check for the existence of the youtube_id
    query = 'SELECT EXISTS(SELECT 1 FROM youtube_videos WHERE youtube_id=? LIMIT 1)'
    cursor.execute(query, (youtube_id,))
    # Fetch the result
    exists = cursor.fetchone()[0]
    conn.close()
    # Return True if exists is 1, otherwise False
    return exists == 1
	
def insert_youtube5m_id(youtube_id):
    conn = sqlite3.connect(db_filename_yt5)
    cursor = conn.cursor()
    # SQL query to insert or ignore the new youtube_id
    insert_query = 'INSERT OR IGNORE INTO youtube_videos (youtube_id) VALUES (?)'
    cursor.execute(insert_query, (youtube_id,))
    conn.commit()
    conn.close()

if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI()
else:
    OPENAI_CLIENT = None

AUGMENT_DESCRIPTIONS = False
AUGMENT_CAPTIONS = False
def get_description(yt: video_utils.YoutubeDL, video_path: str, augmenter) -> str:
    """
    Get / generate the description of a video from the YouTube API.
    
    Miner TODO: Implement logic to get / generate the most relevant and information-rich
    description of a video from the YouTube API.
    """
    description = yt.title
    
    """ caption processing. """
    captions = ""
    if AUGMENT_CAPTIONS:
        frames_dir = f'/workspace/omega_video_frames/{video_path.name.replace(".mp4","").replace("/tmp/","")}'
        #frames_dir = f'/tmp/omega_video_frames/{video_path.name.replace(".mp4","").replace("/tmp/","")}'
        extract_key_frames(video_path.name, frames_dir, 3)
        generate_frames_json(frames_dir)
        captions = generate_captions(frames_dir)
        #if os.path.exists(frames_dir):
            # Use shutil.rmtree() to delete the directory
            #shutil.rmtree(frames_dir)
    
    if AUGMENT_DESCRIPTIONS:
        if yt.description:
            new_description = augmenter.augment_description(yt.description)
        else:
            new_description = augmenter.write_description(yt.title)
        description += f"\n\n{new_description}"
        bt.logging.info(f"Augmented description: {description}")
    elif AUGMENT_CAPTIONS and captions != "":
        new_description = augmenter.augment_description(captions)
        description = new_description
        #description += f"\n\n{new_description}"
        #description += f"\n\n{captions[:5000]}"
        #bt.logging.info(f"Submitting captions description with {len(captions[:5000])} characters.")
        bt.logging.info(f"Augmented caption description: {description}")
    else:
        if yt.description:
            description += f"\n\n{yt.description}"
            
    return description
    
def get_description_from_subtitles(yt, imagebind, clip_path, subtitles, query):
    description = yt.title
    
    if DO_DESCRIPTION_SIMILARITY and len(subtitles) > 0:
        # Preprocess and tokenize subtitles
        preprocessed_subtitles = preprocess_subtitles(subtitles, yt.title, query)

        # Embed all subtitles at once using the embed_text method
        # Since we already tokenized the subtitles, we can directly use the tokens
        subtitle_embeddings = imagebind.embed_text(preprocessed_subtitles).detach()

        # Embed the video once
        with torch.no_grad():  # Ensure no gradients are computed
            video_embedding = imagebind.embed([""], [clip_path]).video[0].detach()

        # Calculate cosine similarity between video embedding and subtitle descriptions
        similarities = F.cosine_similarity(video_embedding.unsqueeze(0), subtitle_embeddings, dim=1)
        # Get the index of the most similar description
        most_similar_index = similarities.argmax().item()
        #bt.logging.info(f"cosign_similarities: {similarities}")

        # check best similarity score against our theshold. If it's good, keep it, otherwise send back "SKIPCLIP" to keep looking for better clips.
        if similarities[most_similar_index] >= SIMILARITY_THRESHOLD:
          # Append the most similar preprocessed subtitle to the description
          description = preprocessed_subtitles[most_similar_index]
          bt.logging.info(f"Submitting description with cosign_similarity of {similarities[most_similar_index]}")
        else:
          description = "SKIPCLIP"
          bt.logging.info(f"cosign_similarity of {similarities[most_similar_index]} is below {SIMILARITY_THRESHOLD} threshold skipping.")

    return description
    
def preprocess_subtitles(subtitles, video_title, query):
    preprocessed_subtitles = []
    for subtitle in subtitles:
        # Tokenize the subtitle by splitting on whitespace
        tokens = subtitle.split()
        # Sort tokens by length, from longest to shortest
        tokens_sorted = sorted(tokens, key=len, reverse=True)
        # Remove tokens with 3 or fewer characters
        tokens_filtered = [token for token in tokens_sorted if len(token) > 9]
        # Remove duplicate words
        unique_tokens = []
        seen_tokens = set()
        for token in tokens_filtered:
            if token not in seen_tokens:
                unique_tokens.append(token)
                seen_tokens.add(token)
        # Rejoin the unique tokens into a preprocessed subtitle string
        preprocessed_subtitle = ' '.join(unique_tokens)
        preprocessed_subtitles.append(preprocessed_subtitle)
        preprocessed_subtitles.append(f"{video_title}\n\n{preprocessed_subtitle}")
        preprocessed_subtitles.append(f"{query}\n\n{preprocessed_subtitle}")
        preprocessed_subtitles.append(f"{video_title}\n\n{query}\n\n{preprocessed_subtitle}")

    preprocessed_subtitles.append(video_title)
    preprocessed_subtitles.append(query)
    preprocessed_subtitles.append(video_title + ". " + query)
    return preprocessed_subtitles
    
def get_optimized_query(original_query, query, video_metas, imagebind):
    # get 5 random augments for the original query
    queries = find_query_augments(original_query, query, limit=5)
    queries.append(original_query)
    queries.append(query)
    # create embeddings of the query permutations
    query_embeddings = imagebind.embed_text(queries).detach().to(imagebind.device)
    video_embeddings = torch.stack([torch.tensor(v.video_emb) for v in video_metas]).to(imagebind.device)
    # Normalize the embeddings to have unit length
    query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
    video_embeddings = F.normalize(video_embeddings, p=2, dim=1)
    
    # Compute the cosine similarity matrix
    similarity_matrix = torch.mm(video_embeddings, query_embeddings.t())

    # Aggregate the similarities for each query across all videos
    # Here we sum the similarities, but you could also use mean()
    query_similarities = similarity_matrix.sum(dim=0)

    # Find the index of the query with the highest total similarity
    most_similar_query_index = query_similarities.argmax().item()
    optimized_query = queries[most_similar_query_index]
    similarity_score = query_similarities[most_similar_query_index].item()

    # Log the most similar query and its aggregated similarity score
    bt.logging.info(f"Submitting query most similar to all videos with a total cosine similarity of {similarity_score}: {optimized_query}")
    
    return optimized_query

def get_relevant_timestamps(query: str, yt: video_utils.YoutubeDL, video_path: str) -> Tuple[int, int]:
    """
    Get the optimal start and end timestamps (in seconds) of a video for ensuring relevance
    to the query.

    Miner TODO: Implement logic to get the optimal start and end timestamps of a video for
    ensuring relevance to the query.
    """
    start_time = 0
    end_time = min(yt.length, MAX_VIDEO_LENGTH)
    return start_time, end_time


async def search_and_embed_videos(original_query: str, query: str, num_videos: int, imagebind: ImageBind, start_function, augmenter) -> List[VideoMetadata]:
    """
    Search YouTube for videos matching the given query and return a list of VideoMetadata objects.

    Args:
        query (str): The query to search for.
        num_videos (int, optional): The number of videos to return.

    Returns:
        List[VideoMetadata]: A list of VideoMetadata objects representing the search results.
    """
    
    # fetch more videos than we need
    results = video_utils.search_videos(query, max_results=int(num_videos * 50))
    # Filter out results that are already in the database
    filtered_results = []
    seen_video_ids = set()
    found_count = 0
    for result in results:
        if result.video_id in seen_video_ids:
            continue
        if (check_youtube_id(result.video_id) and result.length < (6*60)) or check_youtube5m_id(result.video_id):
            found_count += 1
        else:
            filtered_results.append(result)
            seen_video_ids.add(result.video_id)
    bt.logging.info("Found videos in db, skipping", found_count)
    results = filtered_results
    
    video_metas = []
    # fetch random proxy ip
    if len(PROXY_URLS):
        proxy_url = random.choice(PROXY_URLS)
        bt.logging.info("proxy_url:", proxy_url)
    else:
        proxy_url = None
    
    try:
        # take the first N that we need
        for result in results:
            start = time.time()
            
            start = 0
            end = min(result.length, FIVE_MINUTES) # download the first 5 minutes at most
            is_five_minutes = False
            # if the video is longer than 6 minutes, let's try and download from 5 minutes in
            if result.length >= 6*60:
              is_five_minutes = True
              start = FIVE_MINUTES
              end = min(result.length, FIVE_MINUTES*2)
              bt.logging.info(f"Downloading video from 5m to 10m mark")
            
            download_path = video_utils.download_video(
                result.video_id,
                start=start,
                end=end,
                proxy=proxy_url
            )
            if download_path:
                clip_path = None
                try:
                    result.length = video_utils.get_video_duration(download_path.name)  # correct the length
                    bt.logging.info(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    
                    # Grab subtitles of main video
                    subtitles = get_subtitles(download_path)
                    
                    offset = 0 # offset in seconds. i.e. start at 10s in
                    clip_duration = 15  # duration of each clip in seconds
                    number_of_clips = int((result.length-offset) // clip_duration)
                    if number_of_clips == 0:
                      number_of_clips = 1

                    # Loop to create each clip
                    tasks = []
                    for i in range(number_of_clips):
                        task = process_clip(i, number_of_clips, clip_duration, offset, result, download_path, imagebind, subtitles, query, start_function, VALIDATOR_TIMEOUT, TIMEOUT_THRESHOLD, augmenter)
                        tasks.append(task)
                        if len(tasks) >= num_videos:
                            break

                    # Await the tasks and extend video_metas with the new results
                    new_video_metas = await asyncio.gather(*tasks)
                    new_video_metas = [meta for meta in new_video_metas if meta is not None]
                    video_metas.extend(new_video_metas)
                    if len(video_metas) >= num_videos:
                      break
                              
                finally:
                    download_path.close()
                    if clip_path:
                        clip_path.close()
                    insert_youtube_id(result.video_id)
                    if is_five_minutes:
                      insert_youtube5m_id(result.video_id)

            if (time.time() - start_function) > (VALIDATOR_TIMEOUT - TIMEOUT_THRESHOLD):
                bt.logging.info(f"Within {TIMEOUT_THRESHOLD} seconds of VALIDATOR_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - TIMEOUT_THRESHOLD)}s")
                break  
            if len(video_metas) == num_videos:
                break

    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")
        # Log the stack trace
        stack_trace = traceback.format_exc()
        bt.logging.error(stack_trace)

    # run our video_metas through the score checker
    #video_metas = check_video_scores(query, len(video_metas), video_metas)
    
    # find the most optimal query for our videos
    optimized_query = None
    if OPTIMIZE_QUERIES and len(video_metas) > 0:
      optimized_query = get_optimized_query(original_query, query, video_metas, imagebind)
    
    if len(video_metas) > num_videos:
      video_metas = video_metas[:num_videos]
    return video_metas, optimized_query
    
async def process_clip(i, number_of_clips, clip_duration, offset, result, download_path, imagebind, subtitles, query, start_function, VALIDATOR_TIMEOUT, TIMEOUT_THRESHOLD, augmenter):
    clip_path = None
    try:
        start = i * clip_duration + offset
        end = start + clip_duration
        bt.logging.info(f"Creating {clip_duration}s clip of video {result.video_id} -- from {start}s to {end}s")
        
        clip_path = await asyncio.to_thread(
            video_utils.clip_video, download_path.name, start, end
        )
        #clip_path = video_utils.clip_video(download_path.name, start, end)
        
        description = await asyncio.to_thread(
            get_description_from_subtitles, result, imagebind, clip_path, subtitles, query
        )
        #description = get_description_from_subtitles(result, imagebind, clip_path, subtitles, query)
        
        if (time.time() - start_function) > (VALIDATOR_TIMEOUT - TIMEOUT_THRESHOLD):
            bt.logging.info(f"Within {TIMEOUT_THRESHOLD} seconds of VALIDATOR_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - TIMEOUT_THRESHOLD)}s")
            return None
        if description == "SKIPCLIP":
            return None
        if not description or description == "":
            description = get_description(result, download_path, augmenter)
        
        #embeddings = await asyncio.to_thread(
        #    imagebind.embed, [description], [clip_path]
        #)
        embeddings = imagebind.embed([description], [clip_path])
        
        video_meta = VideoMetadata(
            video_id=result.video_id,
            description=description,
            views=result.views,
            start_time=start,
            end_time=end,
            video_emb=embeddings.video[0].tolist(),
            audio_emb=embeddings.audio[0].tolist(),
            description_emb=embeddings.description[0].tolist(),
        )
        return video_meta
    finally:
        if clip_path:
            clip_path.close()    
    
def check_video_scores(query, num_videos, video_metas):
    videos = Videos(query=query, num_videos=num_videos, video_metadata=video_metas)
    video_json = videos.to_serializable_dict(videos)

    url = f'https://dev-validator.api.omega-labs.ai/api/check_score'
    data = requests.get(url, json=video_json).json()
    
    print(data)
    not_unique_indexes = []
    if 'is_unique' in data:
        # Identify the indexes of non-unique items
        not_unique_indexes = [index for index, is_unique in enumerate(data['is_unique']) if not is_unique]
        # Remove corresponding items from 'video_metas'
        video_metas = [meta for index, meta in enumerate(video_metas) if index not in not_unique_indexes]
    
    '''
    description_relevance_score = data['description_relevance_scores'][0]
    query_relevance_score = data['query_relevance_scores'][0]
    novelty_score = data['novelty_score']
    score = data['score']
    '''
    if len(video_metas) < num_videos:
        bt.logging.info(f"Found {(num_videos - len(video_metas))} not unique videos, skipping.")
        
    return video_metas
    

def get_subtitles(video_path: str):
    subtitles = ""
    subtitle_file = video_path.name.replace(".mp4", ".en.srt")
    if not os.path.exists(subtitle_file):
        subtitle_file = video_path.name.replace(".mp4", ".en.vtt")
    if os.path.exists(subtitle_file):
        subtitles = clean_subtitles(subtitle_file)
        os.remove(subtitle_file)
        
    subtitles_list = []
    if subtitles != "":
      subtitles_list = split_string_into_equal_parts(subtitles[:5000], 5)
    
    return subtitles_list

def clean_subtitles(subtitle_file):
    # Read the subtitle file
    with open(subtitle_file, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Remove WEBVTT header and metadata
    content = re.sub(r'WEBVTT.*?\n\n', '', content, flags=re.DOTALL)

    # Remove timestamps and formatting tags
    content = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}(.*?)\n', '', content)
    content = re.sub(r'<\d{2}:\d{2}:\d{2}\.\d{3}>', '', content)
    content = re.sub(r'<c[.\w]*>', '', content)
    content = re.sub(r'</c>', '', content)
    content = re.sub(r'align:start position:\d+% ', '', content)

    # Remove any remaining single newlines and replace double newlines with a single newline
    content = re.sub(r'\n+', '\n', content)

    # Remove any remaining metadata tags
    content = re.sub(r'<[^>]+>', '', content)
    
    content = content.strip()
    
    # Split the string into lines
    lines = content.split('\n')
    # Remove duplicate lines while preserving order
    unique_lines = list(dict.fromkeys(lines))
    # Join the unique lines into a single paragraph
    content = ' '.join(line.strip() for line in unique_lines if line.strip())

    return content
    
def split_string_into_equal_parts(text, num_parts):
    # Calculate the length of each part
    part_length = len(text) // num_parts
    # Initialize an empty list to store the parts
    parts = []
    # Loop to create substrings of equal length
    for i in range(0, len(text), part_length):
        # Add the substring to the list
        parts.append(text[i:i + part_length])
    # Adjust the last part in case of rounding errors
    if len(parts) > num_parts:
        parts[-2] += parts[-1]
        parts = parts[:-1]
    return parts
