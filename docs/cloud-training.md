# Cloud compute and resource management

There is no cloud-provider integration, job scheduler, remote storage client, or cloud-training script in this repository. “Cloud-first” is an operating choice for the user, not an implemented CaptionLens feature.

## Resource behavior in the code

- The API pipeline defaults to CPU (`ProductionCaptionPipeline(device="cpu")`). It loads BLIP and Faster R-CNN when the FastAPI application starts.
- The LoRA script selects CUDA if `torch.cuda.is_available()`; otherwise it uses CPU. It does not select Apple Metal (MPS), configure mixed precision, gradient accumulation, or checkpoint resumption.
- The training code needs the BLIP checkpoint available in the Transformers cache, or network access for the fallback download.
- Inference and training use local process memory. No object-storage upload, remote dataset transfer, or cloud checkpoint sync is implemented.

For a GPU notebook or VM, install the appropriate PyTorch build for that environment, install this project’s requirements, and keep the dataset and checkpoints on storage with sufficient capacity. Check the selected device and available memory before training. These are general deployment steps; the repository does not provision or configure a cloud machine.

Use the [fine-tuning guide](fine-tuning.md) for the manifest and command. The [installation guide](installation.md) describes the local API and Docker setup.
