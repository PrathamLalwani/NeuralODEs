import os
import time
import shutil
from datetime import datetime

# This script monitors the ./experiment_implicit_adams directory for new instances of "logs" and "model.pth" which are then saved with a
# timestamp in their filenames to avoid overwriting.
watch_dir = "./experiment_implicit_adams"
while True:
    model_path = os.path.join(watch_dir, "model.pth")
    if os.path.exists(model_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name_model = f"model_{timestamp}.pth"
        shutil.move(model_path, os.path.join(watch_dir, new_name_model))
        print(f"Renamed model.pth to {new_name_model}")
    time.sleep(3)  # Check for updates over regular intervals