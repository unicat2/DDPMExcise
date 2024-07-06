from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os

class CelebADataset(Dataset):
    def __init__(self):
        super().__init__()
        # self.root = './data/celeba_hq_256'
        self.root = './data_1/Face_CelebA/processed_data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data'
        self.filenames = sorted(os.listdir(self.root))
        self.resolution = (128, 128)

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int):
        path = os.path.join(self.root, self.filenames[index])
        img = Image.open(path)
        pipeline = transforms.Compose([
            transforms.Resize(self.resolution),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: (x - 0.5) * 2)
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        return pipeline(img), 0


def dataloader(dataset_name, batch_size, num_workers=4):
    if dataset_name.lower() == 'celeba_128':
        dataset = CelebADataset()

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def get_img_shape(dataset_name):
    if dataset_name.lower() == 'celeba_128':
        return (3, 128, 128)


if __name__ == '__main__':
    dataset_name = 'celeba_128'
    dataset = CelebADataset()
    data_loader = dataloader(dataset_name, batch_size=64)
    print(len(data_loader.dataset))
    print(get_img_shape(dataset_name))

