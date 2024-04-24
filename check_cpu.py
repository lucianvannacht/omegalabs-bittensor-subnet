import subprocess
import time
import glob
import os
import threading

# Constants for the PM2 task monitoring
PM2_TASK_NAME = 'OMEGAMINER'
CPU_THRESHOLD = 200.0
PROCESS_NAME = 'python3'
TRIP_TRESHOLD = 5

# Constants for the temporary file cleanup
TMP_FILE_PATTERN = '/tmp/tmp*'
FILE_CLEANUP_INTERVAL = 300  # 5 minutes in seconds

def restart_pm2_task(task_name):
    try:
        subprocess.run(['pm2', 'restart', task_name], check=True)
        print(f"Successfully restarted PM2 task: {task_name}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to restart PM2 task: {e}")

def get_cpu_usage_from_top(process_name):
    process = subprocess.Popen(['top', '-bn2'], stdout=subprocess.PIPE)
    output, _ = process.communicate()
    top_output = output.decode('utf-8').split('\n')

    for line in top_output:
        if process_name in line:
            columns = line.split()
            try:
                cpu_usage = float(columns[8])
                return cpu_usage
            except (IndexError, ValueError):
                return None
    return None

def monitor_cpu_and_restart_task():
    trip_count = 0
    while True:
        cpu_usage = get_cpu_usage_from_top(PROCESS_NAME)
        if cpu_usage is not None:
            print(f"Current CPU usage by {PROCESS_NAME}: {cpu_usage}%")
            if cpu_usage >= CPU_THRESHOLD:
                trip_count += 1
                if trip_count >= TRIP_TRESHOLD:
                    print(f"CPU usage by {PROCESS_NAME} above {CPU_THRESHOLD}%, restarting PM2 task...")
                    restart_pm2_task(PM2_TASK_NAME)
                    trip_count = 0
            else:
                trip_count = 0
        else:
            print(f"Could not parse CPU usage for {PROCESS_NAME} from 'top'")
        time.sleep(10)

def delete_tmp_files():
    while True:
        current_time = time.time()
        age_threshold = 5 * 60
        for file_path in glob.glob(TMP_FILE_PATTERN):
            file_mod_time = os.path.getmtime(file_path)
            if current_time - file_mod_time > age_threshold:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
        time.sleep(FILE_CLEANUP_INTERVAL)

if __name__ == "__main__":
    # Create threads for each function
    cpu_monitor_thread = threading.Thread(target=monitor_cpu_and_restart_task)
    file_cleanup_thread = threading.Thread(target=delete_tmp_files)

    # Start the threads
    cpu_monitor_thread.start()
    file_cleanup_thread.start()

    # Join the threads to the main thread
    cpu_monitor_thread.join()
    file_cleanup_thread.join()