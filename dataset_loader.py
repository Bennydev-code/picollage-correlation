import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CorrelationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels.iat[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")
        label = float(self.labels.iat[idx, 1])

        if self.transform is not None:
            image = self.transform(image)

        return image, label
    
if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    csv_path = os.path.expanduser("~/dataset/correlation_assignment/responses.csv")
    img_dir = os.path.expanduser("~/dataset/correlation_assignment/images")
    dataset = CorrelationDataset(csv_path, img_dir)
    print(len(dataset))
