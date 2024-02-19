import kornia.color
from torchvision.utils import save_image
from training.forward import *
from torchvision.transforms import Compose, Lambda
from diffusers import DPMSolverMultistepScheduler




reverse_transform_torch = Compose([
     Lambda(lambda t: (t + 1) / 2),
])

with torch.no_grad():
    def inference_scribs(model, dataloader, channels, image_size, out_path, device, cat):
        create_directory(out_path)
        model.eval()
        scheduler_DPM = DPMSolverMultistepScheduler(beta_schedule='linear', beta_start=1e-4, algorithm_type='dpmsolver++', solver_order=2, num_train_timesteps=1000, thresholding=True)
        scheduler_DPM.set_timesteps(num_inference_steps=100)
        for idx, batch in enumerate(dataloader):
            if idx % 1 == 0:
                batch_size = batch[0].shape[0]
                sketch = batch[0].to(device).to(dtype=torch.float)
                hints = batch[1].to(device).to(dtype=torch.float)

                shape = (batch_size, channels, image_size, image_size)
                torch.manual_seed(2) # Manual seed
                noise = torch.randn(shape, device=device)
                samples = sample_hints(model, noise, sketch, hints, image_size=image_size, batch_size=batch_size,
                                         channels=channels, cat=cat)

                samples_hints = torchvision.utils.make_grid(reverse_transform_torch(hints[:, 0:3, :, :]))
                samples_grid = torchvision.utils.make_grid(reverse_transform_torch(samples[-1]))



                save_image(samples_hints,
                            out_path + 'mask' + str(
                                idx) + '.png')

                save_image(samples_grid,
                               out_path + 'result_' + str(
                                   idx) + '.png')



