"""
Auto-restart bot on file changes (for development)
Usage: python run_dev.py
"""
import subprocess
import sys
import time
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')

BOT_FILE = Path(__file__).parent / "bot.py"
WATCH_FILES = ["bot.py", "payments.py", "features.py", "dome_tracker.py", "api_server.py"]

class RestartHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.start_bot()

    def start_bot(self):
        if self.process:
            print("\n[RESTART] Restarting bot...")
            self.process.terminate()
            self.process.wait()

        print("[START] Starting bot...")
        self.process = subprocess.Popen([sys.executable, str(BOT_FILE)])

    def on_modified(self, event):
        if event.is_directory:
            return

        filename = Path(event.src_path).name
        if filename in WATCH_FILES:
            print(f"\n[CHANGE] Detected change in {filename}")
            time.sleep(0.5)  # Wait for file to finish writing
            self.start_bot()

def main():
    print("[WATCH] Watching for file changes...")
    print(f"   Files: {', '.join(WATCH_FILES)}")
    print("   Press Ctrl+C to stop\n")

    handler = RestartHandler()
    observer = Observer()
    observer.schedule(handler, str(Path(__file__).parent), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")
        observer.stop()
        if handler.process:
            handler.process.terminate()

    observer.join()

if __name__ == "__main__":
    main()
