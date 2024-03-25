# Diffusart - Official implementation

**Diffusart: Enhancing Line Art Colorization with Conditional Diffusion Models** <br>
*Hernan Carrillo, Michaël Clément, Aurélie Bugeau, Edgar Simo-Serra.* <br>
EEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023 <br>
[[Paper](https://openaccess.thecvf.com/content/CVPR2023W/CVFAD/papers/Carrillo_Diffusart_Enhancing_Line_Art_Colorization_With_Conditional_Diffusion_Models_CVPRW_2023_paper.pdf)]

## Reference

Citation:

```latex
@InProceedings{Carrillo_2023_CVPR,
    author    = {Carrillo, Hernan and Cl\'ement, Micha\"el and Bugeau, Aur\'elie and Simo-Serra, Edgar},
    title     = {Diffusart: Enhancing Line Art Colorization With Conditional Diffusion Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2023},
    pages     = {3486-3490}
}
```

## Requirements

- python==3.8
- pytorch==2.1.2
- torchvision==0.16.2

```
conda create -n diffusart python=3.8
conda activate diffusart
pip install -r requirements.txt
```

## Pretrained Model

We uploaded the [pre-trained model](https://drive.google.com/file/d/1q0JUjbPAhd2OFpGPSrMeV8L6eL4FkjOO/view?usp=sharing) to Google drive.

## Training the model [TO DO]
Working..

## Test

```python
python test.py --sketch_dir ./samples/sketch/ --scrib_dir ./samples/scrib/ --out_dir ./samples/results/ --model_path ./checkpoint/diffusart_v1.pth
```
where **--sketch_dir** and **--scrib_dir** are directories that contains the line art and color scribbles images. Both images need to have the same name **(i.e., ./samples/sketch/1118324.jpg  ./samples/scrib/1118324.png)**.

Note: **The color scribbles are 4 dimension images [R,G,B,mask]**

## Abstract

Colorization of line art drawings is an important task in illustration and animation workflows. However, this highly laborious process is mainly done manually, limiting the creative productivity. This paper presents a novel interactive approach for line art colorization using conditional Diffusion Probabilistic Models (DPMs). In our proposed approach, the user provides initial color strokes for colorizing the line art. The strokes are then integrated into the conditional DPM-based colorization process by means of a coupled implicit and explicit conditioning strategy to generates diverse and high-quality colorized images. We evaluate our proposal and show it outperforms existing state-of-the-art approaches using the FID, LPIPS and SSIM metrics.

## Diffusart Framework

Overview of our proposed user-guided line art colorization. The framework is composed of two main components: a denoising model εθ , which learns to generate a denoised image, and an application-specific encoder gθ for extracting user color scribbles information.
<p align="center">
<img src="https://github.com/hernan0930/Diffusart-CVPRW/blob/main/diagrams_img/CVPRW_diagram.png" width=70%>
</p>
