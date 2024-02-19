import torch
import numpy as np
from skimage import io, color
from skimage.transform import resize
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Compose, Resize
import kornia
import matplotlib.pyplot as plt

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        img = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W

        img = img.transpose((2,0,1))
        img = torch.from_numpy(img)

        return img

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image = sample

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = Resize(image, (new_h, new_w))


        return img

    # define image transformations (e.g. using torchvision)
trans = Compose([
    transforms.ToTensor(),
    ])
trans_n = Compose([
    transforms.Lambda(lambda t: (t * 2) - 1)
    ])


class MyData_paper_test(Dataset):
    def __init__(self, sketch_path, scrib_path, size, transform=trans, trans_norm=trans_n):
        self.sketch_path = sketch_path
        self.scrib_path = scrib_path,
        self.transform = transform
        self.trans_norm = trans_norm
        self.size = size

    def __getitem__(self, index):
        f = self.sketch_path[index].split('/')
        s_path = self.sketch_path[index]
        sketch_data = resize(io.imread(s_path), (self.size, self.size))
        hint_in = resize(io.imread(('./samples/scrib/' + f[3][:-4] + '.png')), (self.size, self.size))

        if sketch_data.ndim > 2:
            _, _, c = sketch_data.shape
            if c > 2:
                sketch_data = sketch_data[:, :, 1]



        # Applying transformation (To tensor) and replicating tensor for gray scale images
        if self.transform:
            hint_in = self.transform(hint_in)
            sketch_data = self.transform(np.expand_dims(sketch_data, axis=2))
            hint = self.trans_norm(hint_in[0:3, :, :])
            sketch_data = self.trans_norm(sketch_data)

            hint = torch.cat((hint * hint_in[3:4, :, :], hint_in[3:4, :, :]), 0) # hints: 4 dim [color + mask] [0,4,H,W]

        return sketch_data, hint

    def __len__(self):
        return len(self.sketch_path)
