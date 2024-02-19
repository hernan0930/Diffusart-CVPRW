import torchvision
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import math
from PIL import Image
import os
import torch.distributed as dist
import torch
import tqdm
from diffusers import DDPMScheduler

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def linear_beta_schedule(timesteps, start=1e-6, end=0.02):
    return torch.linspace(start, end, timesteps)

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].
    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.
    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)

def squaredcos_beta_schedule(num_train_timesteps):
    return torch.Tensor(betas_for_alpha_bar(num_train_timesteps))

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t,sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

def load_transformed_dataset(IMG_SIZE):
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = torchvision.datasets.CIFAR10(root=".", download=True,
                                         transform=data_transform)

    test = torchvision.datasets.CIFAR10(root=".", download=True,
                                         transform=data_transform, train=False)
    return torch.utils.data.ConcatDataset([train, test])

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    plt.imshow(reverse_transforms(image))

def tensor_img(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :]
    return reverse_transforms(image)


def show_images(dataset, num_samples=20, cols=4):
    """ Plots some samples from the dataset """
    plt.figure(figsize=(15,15))
    for i, img in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(num_samples/cols + 1, cols, i + 1)
        plt.imshow(img[0])
    plt.show()

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, model, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, model, posterior_variance):
    # Sample noise
    img_size = 32
    img = torch.randn((1, 3, img_size, img_size), device=device)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    T = 50
    stepsize = int(T / num_images)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        img = sample_timestep(img, t, sqrt_one_minus_alphas_cumprod, betas, sqrt_recip_alphas, model, posterior_variance)
        if i % stepsize == 0:
            print( num_images, int(i / stepsize + 1))
            plt.subplot(1, num_images, int(i / stepsize + 1))
            show_tensor_image(img.detach().cpu())
            # tensor_img(img.detach().cpu())
    plt.savefig('test1.png')
    # plt.show()

def extract_inf(a, t, x_shape):
    batch_size = t.shape[0]
    t= t.to(device)
    a = a.to(device)
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(device)

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    t = t.cuda()
    a = a.cuda()
    out = a.gather(-1, t.cuda())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).cuda()

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def save_ckp(state, is_best, checkpoint_dir, best_model_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, f_path)
    if is_best:
        if not os.path.exists(best_model_dir):
            os.makedirs(best_model_dir)
        best_fpath = best_model_dir + '/best_model.pt'
        torch.save(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']

def deprocess(img):
    img = (img + 1) / 2
    return img

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False

    if not dist.is_initialized():
        return False

    return True

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def get_rank():

    if not is_dist_avail_and_initialized():
        return 0

    return dist.get_rank()

def is_main_process():

    return get_rank() == 0

def save_checkpoint_multi(model,  checkpoint_dir):
    f_path = checkpoint_dir + '/checkpoint.pt'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    ckp = model.module.state_dict()
    torch.save(ckp, f_path)

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print
@torch.no_grad()
def ema_update_one(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""


    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())

    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

@torch.no_grad()
def ema_update(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""

    # sync_params(model.parameters())
    # sync_params(averaged_model.parameters())

    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())

    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)


@torch.no_grad()
def ema_update_no(model, averaged_model, decay):
    """Incorporates updated model parameters into an exponential moving averaged
    version of a model. It should be called after each optimizer step."""


    model_params = dict(model.named_parameters())
    averaged_params = dict(averaged_model.named_parameters())

    assert model_params.keys() == averaged_params.keys()

    for name, param in model_params.items():
        averaged_params[name].mul_(decay).add_(param, alpha=1 - decay)

    model_buffers = dict(model.named_buffers())
    averaged_buffers = dict(averaged_model.named_buffers())
    assert model_buffers.keys() == averaged_buffers.keys()

    for name, buf in model_buffers.items():
        averaged_buffers[name].copy_(buf)

def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with torch.no_grad():
            dist.broadcast(p, 0)

def create_directory(directory_path):
    """
    Creates a directory if it does not exist.

    Args:
    directory_path (str): The path of the directory to create.

    Returns:
    bool: True if the directory was created or already exists, False if an error occurred.
    """
    try:
        # Check if the directory already exists
        if not os.path.exists(directory_path):
            # Create the directory, including any intermediate directories
            os.makedirs(directory_path)
            print(f"Directory created: {directory_path}")
        else:
            print(f"Directory already exists: {directory_path}")
        return True
    except Exception as e:
        # An error occurred, print the error message
        print(f"Error creating directory {directory_path}: {e}")
        return False

def sample_DD(model, noise, feat_in, hints, scheduler, cat):
    # samples = []
    # noisy_sample = noise[:, 1::, :, :].to(device, dtype=torch.float)
    # sample = noisy_sample.to(device, dtype=torch.float)

    feat = torch.cat((feat_in.to(device, dtype=torch.float), hints.to(device, dtype=torch.float)), dim=1)

    if cat == True:
        img = noise[:, 1::, :, :].to(device, dtype=torch.float)
    else:
        img = noise.to(device, dtype=torch.float)

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        t = t.to(device, dtype=torch.float)
        t_tensor = torch.tensor([t, ]).to(device, dtype=torch.float)
        if cat == True:
            img = torch.cat((feat_in.to(device, dtype=torch.float), img), dim=1)
        else:
            img = img
            # print(img.shape, feat.shape)

        with torch.no_grad():
            # print(sample.dtype, t.dtype, feat.dtype)
            residual = model(img, feat, t_tensor)

        if cat == True:
            # 2. compute previous image and set x_t -> x_t-1
            img = scheduler.step(residual, t.long(), img[:, 1::, :, :]).prev_sample
        else:
            # 2. compute previous image and set x_t -> x_t-1
            img = scheduler.step(residual, t.long(), img).prev_sample



        # samples.append(img)

    return img


def sample_DDPM(model, noise, feat_in):

    scheduler = DDPMScheduler(beta_start=1e-4, clip_sample=False)
    scheduler.set_timesteps(num_inference_steps=1000)
    # samples = []

    for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
        # 1. predict noise residual
        t = t.cuda()
        t_tensor = torch.tensor([t, ]).cuda()
        img = torch.cat((feat_in.cuda(), noise), dim=1)

        with torch.no_grad():
            residual = model(img, t_tensor)

        # 2. compute previous image and set x_t -> x_t-1
        img = scheduler.step(residual, t.long(), noise).prev_sample

        # samples.append(img)

    return img


def load_image(p):
    return Image.open(p).convert('RGB').resize((512, 512))



