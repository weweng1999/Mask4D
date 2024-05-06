import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

# Path to your TensorFlow event file
event_file_path = '/home/weweng/dev_ws/Mask4D/mask_4d/experiments/mask_4d/lightning_logs/version_30/events.out.tfevents.1714766857.weweng-Oryx-Pro.7375.0'

# Function to read and print the summaries from the event file
def read_tensorflow_events(file_path):
    for e in summary_iterator(file_path):
        for v in e.summary.value:
            print(f"Step: {e.step}, Tag: {v.tag}, Value: {v.simple_value}")

# Read the events
read_tensorflow_events(event_file_path)

