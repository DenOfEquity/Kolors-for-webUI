## (Kwai) Kolors for webui ##
### only tested with Forge2 ###
I don't think there is anything Forge specific here.


### works for me <sup>TM</sup> on 8GB VRAM (GTX1070) ###
#### using controlnet really wants 10GB but this implementation works with 8GB at the cost of performance ####

---
## Install ##
Go to the **Extensions** tab, then **Install from URL**, use the URL for this repository.
### needs updated *diffusers* ###

Easiest way to ensure necessary versions are installed is to edit `requirements_versions.txt` in the webUI folder.
```
diffusers>=0.30.0
accelerate>=0.26.0
```

---
### downloads models on demand - minimum will be ~16.5GB ###

## Model information ##
* https://huggingface.co/Kwai-Kolors/Kolors-diffusers
* https://github.com/Kwai-Kolors/Kolors

---
>[!NOTE]
> if **noUnload** is selected then models are kept in memory; otherwise reloaded for each run. The **unload models** button removes them from memory.


---
<details>
<summary>Change log</summary>

#### 01/08/2025 ####
* support custom models. In `{webui directory}\models\diffusers` create a subdirectory `KolorsCustom`. In that directory, create a subdirectory for the custom model; the directory name will be used as the display name in the newly added **Model** menu. Inside each of those should be the unet file (renamed to *diffusion_pytorch_model.safetensors*) and its `config.json` (copy from extension directory).

#### 23/02/2025 ####
* initial upload. includes IP Adapter (not faceID), controlnets, i2i.


</details>


