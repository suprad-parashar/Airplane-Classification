from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

class AircraftDataset:
    def __init__(self):
        self.IMAGES_PATH = "Data/FGVC Aircraft/data/images"
        self.MANUFACTURER_TRAIN_PATH = "Data/FGVC Aircraft/data/images_manufacturer_train.txt"
        self.VARIANT_TRAIN_PATH = "Data/FGVC Aircraft/data/images_variant_train.txt"
        self.MANUFACTURER_TEST_PATH = "Data/FGVC Aircraft/data/images_manufacturer_test.txt"
        self.VARIANT_TEST_PATH = "Data/FGVC Aircraft/data/images_variant_test.txt"
        self.MANUFACTURER_VAL_PATH = "Data/FGVC Aircraft/data/images_manufacturer_val.txt"
        self.VARIANT_VAL_PATH = "Data/FGVC Aircraft/data/images_variant_val.txt"

        self.classes = {}
        self._train_X, self._train_y = self._get_data(self.MANUFACTURER_TRAIN_PATH, self.VARIANT_TRAIN_PATH)
        self._test_X, self._test_y = self._get_data(self.MANUFACTURER_TEST_PATH, self.VARIANT_TEST_PATH)
        self._val_X, self._val_y = self._get_data(self.MANUFACTURER_VAL_PATH, self.VARIANT_VAL_PATH)

        self._index = 0
        for label in self._train_y + self._test_y + self._val_y:
            if label not in self.classes:
                self.classes[label] = self._index
                self._index += 1

        self.NUM_CLASSES = len(self.classes)
    
    def _get_data(self, manufacturer_data_path, variant_data_path):
        manufacturers = {}
        variants = {}
        with open(manufacturer_data_path, "r") as file:
            for line in file:
                image, manufacturer = line.strip().split(" ", 1)
                manufacturers[image] = manufacturer
        with open(variant_data_path, "r") as file:
            for line in file:
                image, variant = line.strip().split(" ", 1)
                variants[image] = variant
        images = []
        labels = []
        for image in manufacturers:
            images.append(image)
            labels.append(manufacturers[image])# + " " + variants[image])
        return images, labels
    
    class Dataset(Dataset):
        def __init__(self, X, y, classes, IMAGES_PATH="Data/FGVC Aircraft/data/images"):
            self.images = X
            self.labels = y
            self.classes = classes
            self.IMAGES_PATH = IMAGES_PATH
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
            ])
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.transform(cv2.imread(f"{self.IMAGES_PATH}/{self.images[idx]}.jpg")), self.classes[self.labels[idx]]
    
    def get_dataloader(self, type, batch_size=32, shuffle=True):
        if type == "train":
            return DataLoader(self.Dataset(self._train_X, self._train_y, self.classes), batch_size=batch_size, shuffle=shuffle)
        elif type == "test":
            return DataLoader(self.Dataset(self._test_X, self._test_y, self.classes), batch_size=batch_size, shuffle=shuffle)
        elif type == "val":
            return DataLoader(self.Dataset(self._val_X, self._val_y, self.classes), batch_size=batch_size, shuffle=shuffle)
        elif type == "trainval":
            return DataLoader(self.Dataset(self._train_X + self._val_X, self._train_y + self._val_y, self.classes), batch_size=batch_size, shuffle=shuffle)
        else:
            raise ValueError("Invalid type")