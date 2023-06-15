import os
import torch
from PIL import Image
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

def get_default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_images(images, path, **kwargs):
    grid = make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to("cpu").numpy()
    im = Image.fromarray(ndarr)
    im.save(path)

def get(element: torch.Tensor, t: torch.Tensor):
    """
    Get value at index position "t" in "element" and
        reshape it to have the same dimension as a batch of images.
    """
    ele = element.gather(-1, t)
    # return ele.reshape(-1, 1, 1, 1)
    return ele.reshape(-1, 1, 1)

def setup_log_directory(config):
    '''Log and Model checkpoint directory Setup'''

    if os.path.isdir(config.root_log_dir):
        # Get all folders numbers in the root_log_dir
        folder_numbers = [int(folder.replace("version_", "")) for folder in os.listdir(config.root_log_dir)]

        # Find the latest version number present in the log_dir
        last_version_number = max(folder_numbers)

        # New version name
        version_name = f"version_{last_version_number + 1}"

    else:
        version_name = config.log_dir

    # Update the training config default directory
    log_dir        = os.path.join(config.root_log_dir,        version_name)
    checkpoint_dir = os.path.join(config.root_checkpoint_dir, version_name)

    # Create new directory for saving new experiment version
    os.makedirs(log_dir,        exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Logging at: {log_dir}")
    print(f"Model Checkpoint at: {checkpoint_dir}")

    return log_dir, checkpoint_dir

def frames2vid(images, save_path):

    WIDTH = images[0].shape[1]
    HEIGHT = images[0].shape[0]

#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     fourcc = 0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, 25, (WIDTH, HEIGHT))

    # Appending the images to the video one by one
    for image in images:
        video.write(image)

    # Deallocating memories taken for window creation
    # cv2.destroyAllWindows()
    video.release()
    return

def get_dataloader(dataset,
                   batch_size=32,
                   pin_memory=False,
                   shuffle=True,
                   num_workers=0,
                   device="cpu"
                  ):
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            pin_memory=pin_memory,
                            num_workers=num_workers,
                            shuffle=shuffle
                           )
    device_dataloader = DeviceDataLoader(dataloader, device)
    return device_dataloader


class GODFeatureDataset(Dataset):
    def __init__(self, data_path, slice=None, dim=None):
        if 'npy' in data_path:
            data = np.load(data_path)
        elif 'pkl' in data_path:
            with open(data_path, 'rb') as f:
                raw_data = pickle.load(f)
            data = np.zeros((len(raw_data), 512))
            # filepaths = [None] * len(raw_data)
            cnt = 0
            for k, v in raw_data.items():
                data[cnt] = v # v: 512
                # filepaths[cnt] = os.path.join(COCO_unlabel_root, k)
                cnt += 1

        if slice is not None:
            data = data[slice]
        if dim is not None:
            data = self.norm_across_dim(data, dim)
        self.data = torch.from_numpy(data).float()

    def __len__(self):
        return len(self.data)

    def norm_across_dim(self, data, dim):
        self.std = data.std(axis=dim, keepdims=True)
        self.mean = data.mean(axis=dim, keepdims=True)
        return (data - self.mean) / self.std

    def __getitem__(self, idx):
        return self.data[idx].unsqueeze(0) # add ch