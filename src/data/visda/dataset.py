from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class VisDAClassificationDataset(Dataset):
    def __init__(self, root):
        self.root = Path(root)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        self.classes = sorted(
            path.name
            for path in self.root.iterdir()
            if path.is_dir()
        )

        self.class_to_idx = {
            name: index
            for index, name in enumerate(self.classes)
        }

        self.samples = []

        for class_name in self.classes:
            class_dir = self.root / class_name

            for path in class_dir.rglob("*"):
                if path.suffix.lower() in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    self.samples.append(
                        (
                            path,
                            self.class_to_idx[class_name],
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return image, label
