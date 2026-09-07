# Virtual Try-On (VTON) & Ghost Mannequin System

This module is designed for High-End Fashion E-commerce AI processing.

## 1. Modules Loaded
- `diffusers`: For Stable Diffusion based AI Inpainting and Virtual Try-On.
- `transformers`: For Text encoders (CLIP) used to condition the models.
- `accelerate`: To train these models efficiently on multiple GPUs.
- `rembg`: For ultra-sharp AI background removal for garments.
- `albumentations`: For deep learning image augmentations.

## 2. Included Architecture Stubs
- `isbd/vton/architecture.py`: The core VTON Pipeline class structure using `UNet2DConditionModel`.
- `isbd/vton/neck_joint.py`: The `create_neck_joint()` function which removes backgrounds and overlays inner labels/necks on the front garment.

## 3. Future Upgrade Path
When the NVIDIA hardware upgrade is completed:
1. GPU (RTX 30xx/40xx) is hooked up.
2. The placeholder logic in `isbd/vton/architecture.py` will be connected to checkpoints like `IDM-VTON` or a fine-tuned Stable Diffusion.
3. The Docker container will be launched using `--gpus all`.
4. Then this exact structure will run seamlessly for Virtual Dress Trial Rooms.
