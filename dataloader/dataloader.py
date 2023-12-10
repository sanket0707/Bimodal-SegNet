import os
import scipy.io

class DataLoader:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        data = {
            "training": {},
            "testing": {}
        }

        # Loop through training and testing directories
        for phase in ["ESD-1", "ESD-2"]:
            conditions_path = os.path.join(self.dataset_path, phase)
            conditions = [c for c in os.listdir(conditions_path) if os.path.isdir(os.path.join(conditions_path, c))]

            for condition in conditions:
                condition_path = os.path.join(conditions_path, condition)
                data[phase][condition] = {
                    "rgb": {
                        "images": [],
                        "masks": []
                    },
                    "events": {
                        "left": None,
                        "right": None,
                        "events_frame": None,
                        "mask_events_frame": None
                    }
                }

                # Load RGB images and masks
                rgb_images_path = os.path.join(condition_path, "RGB", "images")
                rgb_masks_path = os.path.join(condition_path, "RGB", "masks")
                data[phase][condition]["rgb"]["images"] = self.load_images(rgb_images_path)
                data[phase][condition]["rgb"]["masks"] = self.load_images(rgb_masks_path)

                # Load event data
                events_path = os.path.join(condition_path, "events")
                for file_name in os.listdir(events_path):
                    file_path = os.path.join(events_path, file_name)
                    if file_name.endswith(".mat"):
                        data[phase][condition]["events"][file_name.split('.')[0]] = scipy.io.loadmat(file_path)

        return data

    @staticmethod
    def load_images(directory):
        return [os.path.join(directory, img) for img in os.listdir(directory) if img.endswith(".jpg")]

# Usage
dataset_path = "path_to_your_dataset"  # replace with your dataset path by downloading from https://figshare.com/s/94e5607718545aeb9a4e
loader = DataLoader(dataset_path)
dataset = loader.load_data()

