# this script is partly based on Aladdin Perssons Pix2Pix Implementation 
# for more info check the license file

import torch
import numpy as np
import P2P_config as config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image


class HeightfieldDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = np.array(Image.open(img_path))
        input_image = image[:, :256, :]
        target_image = image[:, 256:, :]

        input_image = Image.fromarray(input_image)
        target_image = Image.fromarray(target_image)
 
        input_image = config.both_transform(input_image)
        target_image = config.both_transform(target_image)
        
        input_image = config.transform_only_input(input_image)
        target_image = config.transform_only_mask(target_image)

        return input_image, target_image


if __name__ == "__main__":
    dataset = HeightfieldDataset(f"{config.ROOT}/04_data/{config.APPLICATION}/{config.VERSION}/train")
    loader = DataLoader(dataset, batch_size=5)
    
    for x, y in loader:

        print(x.shape)
        
        path = f"{config.ROOT}/02_ml/01_hf/casetesting/{config.APPLICATION}/{config.VERSION}/{config.RUN}/datasetDebug/"
        
        import os
        
        if not os.path.isdir(path):
            os.mkdir(path)
            
        save_image(x, os.path.join(path, "x.png"))
        save_image(y, os.path.join(path, "y.png"))
        
        import sys

        sys.exit()