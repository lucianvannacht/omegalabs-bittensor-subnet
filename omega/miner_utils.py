import os
import time
from typing import List, Tuple

import bittensor as bt

from omega.protocol import VideoMetadata
from omega.imagebind_wrapper import ImageBind
from omega.constants import MAX_VIDEO_LENGTH, FIVE_MINUTES, VALIDATOR_TIMEOUT
from omega import video_utils

import asyncio
import traceback
import random

import sqlite3
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

if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    OPENAI_CLIENT = OpenAI()
else:
    OPENAI_CLIENT = None


def get_description(yt: video_utils.YoutubeDL, video_path: str) -> str:
    """
    Get / generate the description of a video from the YouTube API.
    
    Miner TODO: Implement logic to get / generate the most relevant and information-rich
    description of a video from the YouTube API.
    """
    description = yt.title
    if yt.description:
        description += f"\n\n{yt.description}"
    return description


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

# Read proxy URLs from proxies.txt file
PROXY_URLS = []
PROXY_FILENAME = "proxies.txt"
with open(PROXY_FILENAME, 'r') as file:
    for line in file:
        proxy_url = line.strip()
        if proxy_url != "":
            PROXY_URLS.append(proxy_url)

async def search_and_embed_videos(query: str, num_videos: int, imagebind: ImageBind, start_function: float) -> List[VideoMetadata]:
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
    found_count = 0
    for result in results:
        if check_youtube_id(result.video_id):
            found_count += 1
        else:
            filtered_results.append(result)
    print("Found videos in db, skipping", found_count)
    results = filtered_results
    
    video_metas = []
    # fetch random proxy ip
    #proxy_url = random.choice(proxy_urls)
    proxy_url = random.choice(proxy_urls)
    print("proxy_url:", proxy_url)
    try:
        # take the first N that we need
        loop = asyncio.get_event_loop()
        tasks = []
        for result in results:
            if len(tasks) >= num_videos:
                break
            
            end = min(result.length, FIVE_MINUTES)
            task = loop.run_in_executor(None, video_utils.download_video, result.video_id, 0, end, proxy_url)
            tasks.append((result, task))
        
        # Use asyncio.gather to wait for all the futures to complete
        futures = [task[1] for task in tasks]  # Extract only the Future objects
        completed_tasks = await asyncio.gather(*futures, return_exceptions=True)
        
        # Now iterate over the tasks and their completed results
        for (result, _), task_result in zip(tasks, completed_tasks):
            if isinstance(task_result, Exception):
                bt.logging.error(f"An error occurred with video {result.video_id}: {task_result}")
            elif task_result is None:
                bt.logging.error(f"Download returned None for video {result.video_id}, which may indicate a problem.")
            else:
                download_path = task_result
                start = time.time()
                
                try:
                    clip_path = None
                    result.length = video_utils.get_video_duration(download_path.name)
                    bt.logging.info(f"Downloaded video {result.video_id} ({min(result.length, FIVE_MINUTES)}) in {time.time() - start} seconds")
                    start, end = get_relevant_timestamps(query, result, download_path)
                    #description = get_description(result, download_path
                    if (time.time() - start_function) > (VALIDATOR_TIMEOUT - 5):
                    	bt.logging.info(f"Within 10 seconds of Validator_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - 10)}s")
                    	break
                    clip_path = video_utils.clip_video(download_path.name, start, end)
                    if (time.time() - start_function) > (VALIDATOR_TIMEOUT - 5):
                    	bt.logging.info(f"Within 10 seconds of Validator_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - 10)}s")
                    	break                   
                    description = get_description(result, download_path)
                    embeddings = imagebind.embed([description], [clip_path])
                    if (time.time() - start_function) > (VALIDATOR_TIMEOUT - 5):
                    	bt.logging.info(f"Within 10 seconds of Validator_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - 10)}s")
                    	break  
                    video_metas.append(VideoMetadata(
                        video_id=result.video_id,
                        description=description,
                        views=result.views,
                        start_time=start,
                        end_time=end,
                        video_emb=embeddings.video[0].tolist(),
                        audio_emb=embeddings.audio[0].tolist(),
                        description_emb=embeddings.description[0].tolist(),
                    ))
                        
                except Exception as e:
                    bt.logging.error(f"An error occurred while processing video {result.video_id}: {e}")
                    continue
                    
                finally:
                    if download_path:
                        download_path.close()
                    if clip_path:
                        clip_path.close()
                    insert_youtube_id(result.video_id)
                    
                if len(video_metas) == num_videos:
                    break
                if (time.time() - start_function) > (VALIDATOR_TIMEOUT - 5):
                    bt.logging.info(f"Within 10 seconds of Validator_TIMEOUT, breaking loop. {(time.time() - start_function)}s > {(VALIDATOR_TIMEOUT - 10)}s")
                    break  
    except Exception as e:
        bt.logging.error(f"Error searching for videos: {e}")
        # Log the stack trace
        stack_trace = traceback.format_exc()
        bt.logging.error(stack_trace)

    return video_metas
