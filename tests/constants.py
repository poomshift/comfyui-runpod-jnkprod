"""Model folder names shared by the tests."""

# Every folder name ComfyUI v0.36.0 knows, minus custom_nodes (stays in the image)
COMFY_MODEL_FOLDERS = {
    "audio_encoders", "background_removal", "checkpoints", "classifiers",
    "clip_vision", "configs", "controlnet", "datasets", "detection",
    "diffusers", "diffusion_models", "embeddings", "frame_interpolation",
    "geometry_estimation", "gligen", "hypernetworks", "latent_upscale_models",
    "loras", "model_patches", "optical_flow", "photomaker", "style_models",
    "text_encoders", "upscale_models", "vae", "vae_approx",
}

# Registered under folder_paths.models_dir by custom nodes at import:
# Impact-Pack (sams, onnx) and Impact-Subpack (ultralytics/bbox, ultralytics/segm).
CUSTOM_NODE_MODEL_FOLDERS = {"sams", "onnx", "ultralytics/bbox", "ultralytics/segm"}
