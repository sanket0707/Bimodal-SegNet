import numpy as np

def rect(x):
    """ Rectangle function for event aggregation """
    return 1 if abs(x) <= 0.5 else 0

def synchronize_events(event_stream, T, frame_rate):
    """
    Synchronize events with RGB frames.

    :param event_stream: List of tuples (x, y, t, p) representing events
    :param T: Time window for event aggregation
    :param frame_rate: Frame rate of RGB camera
    :return: Synchronized event frames
    """
    # Initialize variables
    max_time = event_stream[-1][2]
    num_frames = int(max_time * frame_rate)
    event_frames = [np.zeros((height, width, 2)) for _ in range(num_frames)]  # Assuming height and width of the frame

    # Aggregate events for each frame
    for x, y, t, p in event_stream:
        frame_idx = int(t * frame_rate)
        if 0 <= frame_idx < num_frames:
            event_frames[frame_idx][x, y, p] += rect((t % T) / T - 0.5)

    return event_frames


