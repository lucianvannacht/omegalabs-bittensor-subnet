import os
import shutil
import sqlite3
from datasets import load_dataset

import time
# Define the interval (1 hours in seconds)
one_hour_in_seconds = 1 * 60 * 60

# Infinite loop to execute every 1 hours
while True:
    try:

        cache_dir = '/workspace/huggingface'  # Specify your cache directory
        # Load the dataset
        existing_ids = set(load_dataset('jondurbin/omega-multimodal-ids', cache_dir=cache_dir)['train']['youtube_id'])

        # Create or open an SQLite database
        conn = sqlite3.connect('youtube.sqlite')
        cur = conn.cursor()

        # Create the table with the given schema
        cur.execute('''
        CREATE TABLE IF NOT EXISTS youtube_videos (
            youtube_id CHAR(30) PRIMARY KEY
        )
        ''')

        # Insert data into the table
        for item in existing_ids:
            # Extract the youtube_id from the item, ensuring it's a string of length <= 30
            #youtube_id = str(item['youtube_id'])[:30]
            youtube_id = str(item)[:30]

            # Inserting the youtube_id into the database
            cur.execute('INSERT OR IGNORE INTO youtube_videos (youtube_id) VALUES (?)',
                        (youtube_id,))

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        print("Data has been successfully inserted into the SQLite database according to the provided schema.")

        # Check if the directory exists
        if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
            # Use shutil.rmtree() to delete the directory
            shutil.rmtree(cache_dir)
            print(f"The directory {cache_dir} has been deleted.")
        else:
            print(f"The directory {cache_dir} does not exist.")
            
    except Exception as e:
        print(f"An error occurred: {e}")

    # Wait for 1 hours before running the script again
    time.sleep(one_hour_in_seconds)
