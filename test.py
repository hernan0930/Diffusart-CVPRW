import argparse
from torch.utils.data import DataLoader
from testing.testing import *
from models.model_coupled_v1 import Unet
from data.data_load import *
import glob
from collections import OrderedDict


device = "cuda:0" if torch.cuda.is_available() else "cpu"

cat = True # Concatenate sketch on input
image_size = 256
channels = 4
batch_size = 1
timesteps= 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sketch_dir', type=str, required=False, default='./samples/sketch/', help='Path to the directory containing line art images.')
    parser.add_argument('--scrib_dir', type=str, required=False, default='./samples/scrib/', help='Path to the directory containing color scribbles images.')
    parser.add_argument('--out_dir', type=str, required=False, default='./samples/results/', help='Path to the directory containing color scribbles images.')
    parser.add_argument('--model_path', type=str, required=False, default='./checkpoint/diffusart_v1.pth', help='Path to the .pth model file.')
    args = parser.parse_args()

    #Reading all images from directories
    sketch_path = glob.glob(args.sketch_dir + '*.jpg')
    scrib_path = glob.glob(args.sketch_dir + '*.png')
    loader_val = MyData_paper_test(sketch_path, scrib_path, size=image_size)

    dataloader_test = DataLoader(loader_val, batch_size=batch_size, num_workers=1, shuffle=False)

    model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=(1, 2,)
    ).to(device)

    print('Entering to inference')

    #Loading the model
    state_dict = torch.load(args.model_path, map_location= device)



    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.to(device)

    inference_scribs(model, dataloader_test, channels, image_size, args.out_dir, device, cat)











