from diffusers.utils import check_min_version
check_min_version("0.30.0")


class KolorsStorage:
    ModuleReload = False
    forgeCanvas = False
    usingGradio4 = False
    pipeTE = None
    pipeTR = None
    lastModel = None
    lastControlNet = None

    lora = None
    lora_scale = 1.0
    loadedLora = False

    lastPrompt = None
    lastNegative = None
    pos_embeds = None
    pos_pooled = None
    neg_embeds = None
    neg_pooled = None
    nul_embeds = None
    nul_pooled = None
    noiseRGBA = [0.0, 0.0, 0.0, 0.0]
    captionToPrompt = False
    sendAccessToken = False
    doneAccessTokenWarning = False

    locked = False     #   for preventing changes to the following volatile state while generating
    karras = False
    randomSeed = True
    noUnload = False
    biasCFG = False
    sharpNoise = False
    i2iAllSteps = False
    centreLatents = False

import gc
import gradio
if int(gradio.__version__[0]) == 4:
    KolorsStorage.usingGradio4 = True
import math
import numpy
import os
import torch
import torchvision.transforms.functional as TF
try:
    from importlib import reload
    KolorsStorage.ModuleReload = True
except:
    KolorsStorage.ModuleReload = False

try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas, canvas_head
    KolorsStorage.forgeCanvas = True
except:
    KolorsStorage.forgeCanvas = False
    canvas_head = ""

from PIL import Image, ImageFilter

##   from webui
from modules import script_callbacks, images, shared
from modules.processing import get_fixed_seed
from modules.shared import opts
from modules.ui_components import ResizeHandleRow, ToolButton
import modules.infotext_utils as parameters_copypaste

##   diffusers / transformers necessary imports
from diffusers import DEISMultistepScheduler, DPMSolverSinglestepScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, UniPCMultistepScheduler, DDPMScheduler
from diffusers import SASolverScheduler, LCMScheduler
from diffusers.utils.torch_utils import randn_tensor
from diffusers.utils import logging
from transformers import CLIPVisionModelWithProjection
from scripts.kolors_controlnet import ControlNetModel

##  for Florence-2
from transformers import AutoProcessor, AutoModelForCausalLM 
##  for SuperPrompt
from transformers import T5TokenizerFast, T5ForConditionalGeneration

##   my extras
import customStylesListKolors as styles
import scripts.kolors_pipeline as pipeline



# modules/processing.py - don't use ',', '\n', ':' in values
def create_infotext(sampler, positive_prompt, negative_prompt, guidance_scale, guidance_rescale, steps, seed, width, height, PAG_scale, PAG_adapt, loraSettings, controlNetSettings):
    bCFG = ", biasCorrectionCFG: enabled" if KolorsStorage.biasCFG == True else ""
    karras = " : Karras" if KolorsStorage.karras == True else ""
    generation_params = {
        "Steps": steps,
        "CFG scale": f"{guidance_scale}",
        "CFG rescale": f"{guidance_rescale}",
        "Seed": seed,
        "Size": f"{width}x{height}",
        "PAG": f"{PAG_scale} ({PAG_adapt})",
        "Sampler": f"{sampler}{karras}",
        "LoRA"          :   loraSettings,
        "controlNet"    :   controlNetSettings,
    }
#add i2i marker?
    prompt_text = f"{positive_prompt}\n"
    if negative_prompt != "":
        prompt_text += (f"Negative prompt: {negative_prompt}\n")
    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])
    noise_text = f", Initial noise: {KolorsStorage.noiseRGBA}" if KolorsStorage.noiseRGBA[3] != 0.0 else ""

    return f"{prompt_text}{generation_params_text}{noise_text}{bCFG}, Model (Kolors)"

def predict(positive_prompt, negative_prompt, sampler, width, height, guidance_scale, guidance_rescale, num_steps, sampling_seed, num_images, i2iSource, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, maskSource, maskBlur, maskCutOff, IPASource, IPAType, IPAScale, controlNetImage, controlNet, controlNetStrength, controlNetStart, controlNetEnd,  *args):
 
    logging.set_verbosity(logging.ERROR)        #   diffusers and transformers both enjoy spamming the console with useless info
 
    access_token = 0
    if KolorsStorage.sendAccessToken == True:
        try:
            with open('huggingface_access_token.txt', 'r') as file:
                access_token = file.read().strip()
        except:
            if KolorsStorage.doneAccessTokenWarning == False:
                print ("Kolors: couldn't load 'huggingface_access_token.txt' from the webui directory. Will not be able to download/update gated models. Local cache will work.")
                KolorsStorage.doneAccessTokenWarning = True

    torch.set_grad_enabled(False)


    dtype = torch.float16
    variant = "fp16"
    
    if style != 0:
        positive_prompt = styles.styles_list[style][1].replace("{prompt}", positive_prompt)
        negative_prompt = negative_prompt + styles.styles_list[style][2]

    if PAG_scale > 0.0:
        guidance_rescale = 0.0

    ####    check img2img
    if i2iSource == None:
        maskType = 0
        i2iDenoise = 1
    
    if maskSource == None:
        maskType = 0

    match maskType:
        case 0:     #   'none'
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0
        case 1:
            if KolorsStorage.forgeCanvas: #  'inpaint mask'
                maskSource = maskSource.getchannel('A').convert('L')#.convert("RGB")#.getchannel('R').convert('L')
            else:                       #   'drawn'
                maskSource = maskSource['layers'][0]  if KolorsStorage.usingGradio4 else maskSource['mask']
        case 2:
            if KolorsStorage.forgeCanvas: #   sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                maskSource = None
                maskBlur = 0
                maskCutOff = 1.0
            else:                       #   'image'
                maskSource = maskSource['background'] if KolorsStorage.usingGradio4 else maskSource['image']
        case 3:
            if KolorsStorage.forgeCanvas: #   inpaint sketch
                i2iSource = Image.alpha_composite(i2iSource, maskSource)
                mask = maskSource.getchannel('A').convert('L')
                short_side = min(mask.size)
                dilation_size = int(0.015 * short_side) * 2 + 1
                mask = mask.filter(ImageFilter.MaxFilter(dilation_size))
                maskSource = mask.point(lambda v: 255 if v > 0 else 0)
                maskCutoff = 0.0
            else:                       #   'composite'
                maskSource = maskSource['composite']  if KolorsStorage.usingGradio4 else maskSource['image']
        case _:
            maskSource = None
            maskBlur = 0
            maskCutOff = 1.0

    if i2iSource:
        if KolorsStorage.i2iAllSteps == True:
            num_steps = int(num_steps / i2iDenoise)

        if KolorsStorage.forgeCanvas:
            i2iSource = i2iSource.convert('RGB')

    if maskBlur > 0:
        dilation_size = maskBlur * 2 + 1
        maskSource = TF.gaussian_blur(maskSource.filter(ImageFilter.MaxFilter(dilation_size)), dilation_size)
    ####    end check img2img

    ####    controlnet
    useControlNet = None

    match controlNet:
        case 1:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'Kwai-Kolors/Kolors-ControlNet-Canny'
        case 2:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'Kwai-Kolors/Kolors-ControlNet-Depth'
        case 3:
            if controlNetImage and controlNetStrength > 0.0:
                useControlNet = 'Kwai-Kolors/Kolors-ControlNet-Pose'
        case _:
            controlNetStrength = 0.0
            controlNetImage = None

    if useControlNet:
        controlNetImage = controlNetImage.convert('RGB')#.resize((width, height))
    ####    end controlnet
    
    if IPAType == "(None)":
        IPASource = None
 
    fixed_seed = get_fixed_seed(-1 if KolorsStorage.randomSeed else sampling_seed)

    ####    text encoding
    calcEmbeds = (KolorsStorage.lastPrompt   != positive_prompt) or \
                 (KolorsStorage.lastNegative != negative_prompt) or \
                 (KolorsStorage.pos_embeds is None) or \
                 (KolorsStorage.neg_embeds is None) or \
                 (KolorsStorage.nul_embeds is None)
    if calcEmbeds:
        ####    setup pipe for text encoding
        if KolorsStorage.pipeTE == None:
            KolorsStorage.pipeTE = pipeline.KolorsPipeline_DoE.from_pretrained(
                "Kwai-Kolors/Kolors-diffusers",
                vae=None,
                unet=None,
                scheduler=None,
                variant=variant,
                torch_dtype=dtype
            )

        # KolorsStorage.pipeTE.to('cuda')
        KolorsStorage.pipeTE.enable_sequential_cpu_offload()

        print ("Kolors: encoding prompt ...", end="\r", flush=True)
        if KolorsStorage.lastPrompt != positive_prompt or KolorsStorage.pos_embeds is None:
            pos_embeds, pos_pooled = KolorsStorage.pipeTE.encode_prompt(
                positive_prompt,
            )
            KolorsStorage.pos_embeds    = pos_embeds.to('cuda')
            KolorsStorage.pos_pooled    = pos_pooled.to('cuda')
            del pos_embeds, pos_pooled
            KolorsStorage.lastPrompt = positive_prompt

        if KolorsStorage.lastNegative != negative_prompt or KolorsStorage.neg_embeds is None:
            neg_embeds, neg_pooled = KolorsStorage.pipeTE.encode_prompt(
                negative_prompt,
            )
            KolorsStorage.neg_embeds    = neg_embeds.to('cuda')
            KolorsStorage.neg_pooled    = neg_pooled.to('cuda')
            del neg_embeds, neg_pooled
            KolorsStorage.lastNegative = negative_prompt

        if KolorsStorage.nul_embeds is None:
            nul_embeds, nul_pooled = KolorsStorage.pipeTE.encode_prompt(
                "",
            )
            KolorsStorage.nul_embeds    = nul_embeds.to('cuda')
            KolorsStorage.nul_pooled    = nul_pooled.to('cuda')
            del nul_embeds, nul_pooled

        print ("Kolors: encoding prompt ... done")
    else:
        print ("Kolors: Skipping tokenizer, text_encoder.")

    if KolorsStorage.noUnload:
        pass
    else:
        KolorsStorage.pipeTE = None
        gc.collect()
        torch.cuda.empty_cache()
    ####    end text encoding

    ####    setup pipe for transformer + VAE
    if KolorsStorage.pipeTR == None:
        KolorsStorage.pipeTR = pipeline.KolorsPipeline_DoE.from_pretrained(
            "Kwai-Kolors/Kolors-diffusers",
            tokenizer=None,
            text_encoder=None,
            controlnet=None,
            variant=variant,
            torch_dtype=dtype,
        )
        #image_encoder
        #feature_extractor - for instant id?

    if useControlNet:
        if useControlNet != KolorsStorage.lastControlNet:
            KolorsStorage.pipeTR.controlnet=ControlNetModel.from_pretrained(
                useControlNet, torch_dtype=torch.float16,
            ).to('cuda')
            KolorsStorage.lastControlNet = useControlNet
    else:
        KolorsStorage.pipeTR.controlnet = None
        KolorsStorage.lastControlNet = useControlNet


    if IPASource is not None and IPAType != "(None)":
        KolorsStorage.pipeTR.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                                                    "Kwai-Kolors/Kolors-IP-Adapter-Plus",
                                                    subfolder="image_encoder",
                                                    low_cpu_mem_usage=True,
                                                    torch_dtype=torch.float16,
                                                    revision="refs/pr/4",
                                                )
        KolorsStorage.pipeTR.load_ip_adapter(
            "Kwai-Kolors/Kolors-IP-Adapter-Plus",
            subfolder="",
            weight_name="ip_adapter_plus_general.safetensors",
            revision="refs/pr/4",
            image_encoder_folder=None,
        )
        KolorsStorage.pipeTR.set_ip_adapter_scale([IPAScale])
        
        #KolorsStorage.pipeTR.set_face_fidelity_scale(scale)
    else:
        KolorsStorage.pipeTR.image_encoder = None
        KolorsStorage.pipeTR.feature_extractor = None

    KolorsStorage.pipeTR.enable_model_cpu_offload()

    ####    end setup pipe for unet + VAE


    # KolorsStorage.pipeTR.scheduler = scheduler

#   load in LoRA
    if KolorsStorage.lora and KolorsStorage.lora != "(None)" and KolorsStorage.lora_scale != 0.0:
        lorapath = ".//models/diffusers//KolorsLora//"
        loraname = KolorsStorage.lora + ".safetensors"
        try:
            KolorsStorage.pipeTR.load_lora_weights(lorapath+loraname, adapter_name="lora")
            KolorsStorage.loadedLora = True
        except:
            print ("Kolors: failed LoRA: " + loraname)
            #   no reason to abort, just carry on without LoRA

    shape = (
        num_images,
        KolorsStorage.pipeTR.unet.config.in_channels,
        int(height) // KolorsStorage.pipeTR.vae_scale_factor,
        int(width) // KolorsStorage.pipeTR.vae_scale_factor,
    )

    #   always generate the noise here
    generator = [torch.Generator(device='cpu').manual_seed(fixed_seed+i) for i in range(num_images)]
    latents = randn_tensor(shape, generator=generator).to('cuda')
    
    if KolorsStorage.sharpNoise:
        minDim = 1 + (min(latents.size(2), latents.size(3)) // 2)
        for b in range(len(latents)):
            blurred = TF.gaussian_blur(latents[b], minDim)
            latents[b] = 1.02*latents[b] - 0.02*blurred
    
    
    #regen the generator to minimise differences between single/batch - might still be different - batch processing could use different pytorch kernels
    del generator
    generator = torch.Generator(device='cpu').manual_seed(14641)

    KolorsStorage.pipeTR.vae.to(dtype=torch.float32)

    #   colour the initial noise
    if KolorsStorage.noiseRGBA[3] != 0.0:
        nr = KolorsStorage.noiseRGBA[0]
        ng = KolorsStorage.noiseRGBA[1]
        nb = KolorsStorage.noiseRGBA[2]

        imageR = torch.tensor(numpy.full((8,8), (nr+nr-1.0), dtype=numpy.float32))
        imageG = torch.tensor(numpy.full((8,8), (ng+ng-1.0), dtype=numpy.float32))
        imageB = torch.tensor(numpy.full((8,8), (nb+nb-1.0), dtype=numpy.float32))
        image = torch.stack((imageR, imageG, imageB), dim=0).unsqueeze(0)
        
        image = KolorsStorage.pipeTR.image_processor.preprocess(image).to('cuda')
        image_latents = KolorsStorage.pipeTR.vae.encode(image).latent_dist.sample()
        image_latents *= KolorsStorage.pipeTR.vae.config.scaling_factor
        image_latents = image_latents.to(latents.dtype)
        image_latents = image_latents.repeat(num_images, 1, latents.size(2), latents.size(3))

#        latents += image_latents * KolorsStorage.noiseRGBA[3]
        torch.lerp (latents, image_latents, KolorsStorage.noiseRGBA[3], out=latents)

        # NoiseScheduler = KolorsStorage.pipeTR.scheduler
        # ts = torch.tensor([int(1000 * (1.0-(0.1*KolorsStorage.noiseRGBA[3]))) - 1], device='cpu')
        # ts = ts[:1].repeat(num_images)
        # latents = NoiseScheduler.add_noise(image_latents, latents, ts)

        del imageR, imageG, imageB, image, image_latents#, NoiseScheduler
    #   end: colour the initial noise



    schedulerConfig = dict(KolorsStorage.pipeTR.scheduler.config)
    schedulerConfig['use_karras_sigmas'] = KolorsStorage.karras
    schedulerConfig.pop('algorithm_type', None) 
    
    if sampler == 'DDPM':
        KolorsStorage.pipeTR.scheduler = DDPMScheduler.from_config(schedulerConfig)
    elif sampler == 'DEIS':
        KolorsStorage.pipeTR.scheduler = DEISMultistepScheduler.from_config(schedulerConfig)
    elif sampler == 'DPM++ 2M':
        KolorsStorage.pipeTR.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif sampler == "DPM++ 2M SDE":
        schedulerConfig['algorithm_type'] = 'sde-dpmsolver++'
        KolorsStorage.pipeTR.scheduler = DPMSolverMultistepScheduler.from_config(schedulerConfig)
    elif sampler == 'DPM':
        KolorsStorage.pipeTR.scheduler = DPMSolverSinglestepScheduler.from_config(schedulerConfig)
    elif sampler == 'DPM SDE':
        KolorsStorage.pipeTR.scheduler = DPMSolverSDEScheduler.from_config(schedulerConfig)
    elif sampler == 'Euler':
        KolorsStorage.pipeTR.scheduler = EulerDiscreteScheduler.from_config(schedulerConfig)
    elif sampler == 'Euler A':
        KolorsStorage.pipeTR.scheduler = EulerAncestralDiscreteScheduler.from_config(schedulerConfig)
    elif sampler == 'LCM':
        KolorsStorage.pipeTR.scheduler = LCMScheduler.from_config(schedulerConfig)
    elif sampler == "SA-solver":
        schedulerConfig['algorithm_type'] = 'data_prediction'
        KolorsStorage.pipeTR.scheduler = SASolverScheduler.from_config(schedulerConfig)
    elif sampler == 'UniPC':
        KolorsStorage.pipeTR.scheduler = UniPCMultistepScheduler.from_config(schedulerConfig)
    else:
        KolorsStorage.pipeTR.scheduler = DDPMScheduler.from_config(schedulerConfig)



    timesteps = None

#    if useCustomTimeSteps:
#    timesteps = [999, 845, 730, 587, 443, 310, 193, 116, 53, 13]    #   AYS sdXL
    #loglin interpolate to number of steps

    KolorsStorage.pipeTR.vae.to('cpu')
    # gc.collect()
    # torch.cuda.empty_cache()

    with torch.inference_mode():
        output = KolorsStorage.pipeTR(
            generator                       = generator,
            latents                         = latents.to(dtype),   #   initial noise, possibly with colour biasing

            image                           = i2iSource,
            mask_image                      = maskSource,
            strength                        = i2iDenoise,
            mask_cutoff                     = maskCutOff,

            num_inference_steps             = num_steps,
            num_images_per_prompt           = num_images,
            height                          = height,
            width                           = width,
            guidance_scale                  = guidance_scale,
            guidance_rescale                = guidance_rescale,
            prompt_embeds                   = KolorsStorage.pos_embeds,
            negative_prompt_embeds          = KolorsStorage.neg_embeds,
            nul_prompt_embeds               = KolorsStorage.nul_embeds,
            pooled_prompt_embeds            = KolorsStorage.pos_pooled,
            negative_pooled_prompt_embeds   = KolorsStorage.neg_pooled,
            nul_pooled_prompt_embeds        = KolorsStorage.nul_pooled,
            
            do_bias_CFG                     = KolorsStorage.biasCFG,
            
            pag_scale                       = PAG_scale,
            pag_adaptive_scale              = PAG_adapt,
            
            ip_adapter_image                = IPASource,
            
            control_image                   = controlNetImage if useControlNet else None, 
            controlnet_conditioning_scale   = controlNetStrength,  
            control_start                   = controlNetStart,
            control_end                     = controlNetEnd,

            
            centre_latents                  = KolorsStorage.centreLatents,
            cross_attention_kwargs          = {"scale": KolorsStorage.lora_scale}
        ).images

    if KolorsStorage.noUnload:
        if KolorsStorage.loadedLora == True:
            KolorsStorage.pipeTR.unload_lora_weights()
            KolorsStorage.loadedLora = False
        KolorsStorage.pipeTR.unet.to('cpu')
    else:
        KolorsStorage.pipeTR.unet = None
        KolorsStorage.lastModel = None
        KolorsStorage.lastControlNet = None

    del generator, latents

    gc.collect()
    torch.cuda.empty_cache()

    KolorsStorage.pipeTR.vae.to('cuda')

    needs_upcasting = KolorsStorage.pipeTR.vae.dtype == torch.float16 and KolorsStorage.pipeTR.vae.config.force_upcast

    if needs_upcasting:
        KolorsStorage.pipeTR.upcast_vae()
        output = output.to(next(iter(KolorsStorage.pipeTR.vae.post_quant_conv.parameters())).dtype)
    elif output.dtype != KolorsStorage.pipeTR.vae.dtype:
        if torch.backends.mps.is_available():
            # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
            KolorsStorage.pipeTR.vae = KolorsStorage.pipeTR.vae.to(output.dtype)

    if KolorsStorage.lora != "(None)" and KolorsStorage.lora_scale != 0.0:
        loraSettings = KolorsStorage.lora + f" ({KolorsStorage.lora_scale})"
    else:
        loraSettings = None

    if useControlNet != None:
        useControlNet += f" strength: {controlNetStrength}; step range: {controlNetStart}-{controlNetEnd}"

    original_samples_filename_pattern = opts.samples_filename_pattern
    opts.samples_filename_pattern = "Kolors_[datetime]"
    results = []
    total = len(output)
    for i in range (total):
        print (f'Kolors: VAE: {i+1} of {total}', end='\r', flush=True)
        latent = output[i:i+1].to(KolorsStorage.pipeTR.vae.dtype)
        image = KolorsStorage.pipeTR.vae.decode(latent / KolorsStorage.pipeTR.vae.config.scaling_factor, return_dict=False)[0]
        image = KolorsStorage.pipeTR.image_processor.postprocess(image, output_type="pil")[0]

        info=create_infotext(
            sampler,
            positive_prompt, negative_prompt,
            guidance_scale, guidance_rescale, 
            num_steps, 
            fixed_seed + i, 
            width, height,
            PAG_scale, PAG_adapt,
            loraSettings, useControlNet)

        if maskType > 0 and maskSource is not None:
            # i2iSource = i2iSource.convert('RGB')
            # image = image.convert('RGB')
            image = Image.composite(image, i2iSource, maskSource)

        results.append((image, info))
        
        images.save_image(
            image,
            opts.outdir_samples or opts.outdir_txt2img_samples,
            "",
            fixed_seed + i,
            positive_prompt,
            opts.samples_format,
            info
        )
    print ('Kolors: VAE: done  ')
    opts.samples_filename_pattern = original_samples_filename_pattern

    del output

    if KolorsStorage.noUnload:
        KolorsStorage.pipeTR.vae.to('cpu')
    else:
        KolorsStorage.pipeTR = None
        KolorsStorage.lastModel = None

    gc.collect()
    torch.cuda.empty_cache()

    return fixed_seed, gradio.Button.update(interactive=True), results


def on_ui_tabs():
    if KolorsStorage.ModuleReload:
        reload(styles)
        reload(pipeline)
    
    defaultWidth = 1024
    defaultHeight = 1024
 
    def buildLoRAList ():
        loras = ["(None)"]
        
        import glob
        customLoRA = glob.glob(".\models\diffusers\KolorsLora\*.safetensors")

        for i in customLoRA:
            filename = i.split('\\')[-1]
            loras.append(filename[0:-12])

        return loras

    loras = buildLoRAList ()

    def refreshLoRAs ():
        loras = buildLoRAList ()
        return gradio.Dropdown.update(choices=loras)
 
    def getGalleryIndex (index):
        if index < 0:
            index = 0
        return index

    def getGalleryText (gallery, index, seed):
        if gallery:
            return gallery[index][1], seed+index
        else:
            return "", seed+index

    def i2iSetDimensions (image, w, h):
        if image is not None:
            w = image.size[0]
            h = image.size[1]
        return [w, h]

    def i2iMakeCaptions (image, originalPrompt):
        if image == None:
            return originalPrompt
        image = image.convert('RGB')

        model = AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-base', 
                                                         attn_implementation="sdpa", 
                                                         torch_dtype=torch.float16, 
                                                         trust_remote_code=True).to('cuda')
        processor = AutoProcessor.from_pretrained('microsoft/Florence-2-base', #-large
                                                  torch_dtype=torch.float32, 
                                                  trust_remote_code=True)

        result = ''
        prompts = ['<MORE_DETAILED_CAPTION>']

        for p in prompts:
            inputs = processor(text=p, images=image.convert("RGB"), return_tensors="pt")
            inputs.to('cuda').to(torch.float16)
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
            del inputs
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            del generated_ids
            parsed_answer = processor.post_process_generation(generated_text, task=p, image_size=(image.width, image.height))
            del generated_text
            print (parsed_answer)
            result += parsed_answer[p]
            del parsed_answer
            if p != prompts[-1]:
                result += ' | \n'

        del model, processor

        if KolorsStorage.captionToPrompt:
            return result
        else:
            return originalPrompt

    def i2iImageFromGallery (gallery, index):
        try:
            if KolorsStorage.usingGradio4:
                newImage = gallery[index][0]
                return newImage
            else:
                newImage = gallery[index][0]['name'].rsplit('?', 1)[0]
                return newImage
        except:
            return None

    def toggleC2P ():
        KolorsStorage.captionToPrompt ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][KolorsStorage.captionToPrompt])

    def toggleAccess ():
        KolorsStorage.sendAccessToken ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][KolorsStorage.sendAccessToken])

    #   these are volatile state, should not be changed during generation
    def toggleNU ():
        if not KolorsStorage.locked:
            KolorsStorage.noUnload ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][KolorsStorage.noUnload])

    def unloadM ():
        if not KolorsStorage.locked:
            KolorsStorage.pipeTE = None
            KolorsStorage.pipeTR = None
            KolorsStorage.lastModel = None
            shared.SuperPrompt_tokenizer = None
            shared.SuperPrompt_model = None

            gc.collect()
            torch.cuda.empty_cache()
        else:
            gradio.Info('Unable to unload models while using them.')

    def toggleRandom ():
        KolorsStorage.randomSeed ^= True
        return gradio.Button.update(variant='primary' if KolorsStorage.randomSeed == True else 'secondary')

    def toggleKarras ():
        if not KolorsStorage.locked:
            KolorsStorage.karras ^= True
        return gradio.Button.update(variant='primary' if KolorsStorage.karras == True else 'secondary',
                                value='\U0001D40A' if KolorsStorage.karras == True else '\U0001D542')

    def toggleBiasCFG ():
        if not KolorsStorage.locked:
            KolorsStorage.biasCFG ^= True
        return gradio.Button.update(variant='primary' if KolorsStorage.biasCFG == True else 'secondary')


    def toggleAS ():
        if not KolorsStorage.locked:
            KolorsStorage.i2iAllSteps ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][KolorsStorage.i2iAllSteps])

    def toggleCL ():
        if not KolorsStorage.locked:
            KolorsStorage.centreLatents ^= True
        return gradio.Button.update(variant=['secondary', 'primary'][KolorsStorage.centreLatents],
                                value=['\u29BE', '\u29BF'][KolorsStorage.centreLatents])

    def toggleSP ():
        if not KolorsStorage.locked:
            return gradio.Button.update(variant='primary')
    def superPrompt (prompt, seed):
        tokenizer = getattr (shared, 'SuperPrompt_tokenizer', None)
        superprompt = getattr (shared, 'SuperPrompt_model', None)
        if tokenizer is None:
            tokenizer = T5TokenizerFast.from_pretrained(
                'roborovski/superprompt-v1',
            )
            shared.SuperPrompt_tokenizer = tokenizer
        if superprompt is None:
            superprompt = T5ForConditionalGeneration.from_pretrained(
                'roborovski/superprompt-v1',
                device_map='auto',
                torch_dtype=torch.float16
            )
            shared.SuperPrompt_model = superprompt
            print("SuperPrompt-v1 model loaded successfully.")
            if torch.cuda.is_available():
                superprompt.to('cuda')

        torch.manual_seed(get_fixed_seed(seed))
        device = superprompt.device
        systemprompt1 = "Expand the following prompt to add more detail: "
        
        input_ids = tokenizer(systemprompt1 + prompt, return_tensors="pt").input_ids.to(device)
        outputs = superprompt.generate(input_ids, max_new_tokens=256, repetition_penalty=1.2, do_sample=True)
        dirty_text = tokenizer.decode(outputs[0])
        result = dirty_text.replace("<pad>", "").replace("</s>", "").strip()
        
        return gradio.Button.update(variant='secondary'), result



    def toggleGenerate (R, G, B, A, lora, scale):
        KolorsStorage.noiseRGBA = [R, G, B, A]
        KolorsStorage.lora = lora
        KolorsStorage.lora_scale = scale# if lora != "(None)" else 1.0
        KolorsStorage.locked = True
        return gradio.Button.update(value='...', variant='secondary', interactive=False), gradio.Button.update(interactive=False)

    def afterGenerate ():
        KolorsStorage.locked = False
        return gradio.Button.update(value='Generate', variant='primary', interactive=True)

    schedulerList = ["default", "DDPM", "DEIS", "DPM++ 2M", "DPM++ 2M SDE", "DPM", "DPM SDE",
                     "Euler", "Euler A", "LCM", "SA-solver", "UniPC", ]

    def parsePrompt (positive, negative, sampler, width, height, seed, steps, cfg, rescale, nr, ng, nb, ns, PAG_scale, PAG_adapt, loraName, loraScale):
        p = positive.split('\n')
        lineCount = len(p)

        negative = ''
        
        if "Prompt" != p[0] and "Prompt: " != p[0][0:8]:               #   civitAI style special case
            positive = p[0]
            l = 1
            while (l < lineCount) and not (p[l][0:17] == "Negative prompt: " or p[l][0:7] == "Steps: " or p[l][0:6] == "Size: "):
                if p[l] != '':
                    positive += '\n' + p[l]
                l += 1
        
        for l in range(lineCount):
            if "Prompt" == p[l][0:6]:
                if ": " == p[l][6:8]:                                   #   mine (old)
                    positive = str(p[l][8:])
                    c = 1
                elif "Prompt" == p[l] and (l+1 < lineCount):            #   webUI
                    positive = p[l+1]
                    c = 2
                else:
                    continue

                while (l+c < lineCount) and not (p[l+c][0:10] == "Negative: " or p[l+c][0:15] == "Negative Prompt" or p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        positive += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Negative" == p[l][0:8]:
                if ": " == p[l][8:10]:                                  #   mine (old)
                    negative = str(p[l][10:])
                    c = 1
                elif " prompt: " == p[l][8:17]:                         #   civitAI
                    negative = str(p[l][17:])
                    c = 1
                elif " Prompt" == p[l][8:15] and (l+1 < lineCount):     #   webUI
                    negative = p[l+1]
                    c = 2
                else:
                    continue
                
                while (l+c < lineCount) and not (p[l+c] == "Params" or p[l+c][0:7] == "Steps: " or p[l+c][0:6] == "Size: "):
                    if p[l+c] != '':
                        negative += '\n' + p[l+c]
                    c += 1
                l += 1

            elif "Initial noise: " == str(p[l][0:15]):
                noiseRGBA = str(p[l][16:-1]).split(',')
                nr = float(noiseRGBA[0])
                ng = float(noiseRGBA[1])
                nb = float(noiseRGBA[2])
                ns = float(noiseRGBA[3])
            else:
                params = p[l].split(',')
                for k in range(len(params)):
                    pairs = params[k].strip().split(' ')
                    match pairs[0]:
                        case "Size:":
                            size = pairs[1].split('x')
                            width = 16 * ((int(size[0]) + 8) // 16)
                            height = 16 * ((int(size[1]) + 8) // 16)
                        case "Seed:":
                            seed = int(pairs[1])
                        case "Steps(Prior/Decoder):":
                            steps = str(pairs[1]).split('/')
                            steps = int(steps[0])
                        case "Steps:":
                            steps = int(pairs[1])
                        case "CFG":
                            if "scale:" == pairs[1]:
                                cfg = float(pairs[2])
                            elif "rescale:" == pairs[1]:
                                rescale = float(pairs[2])
                            else:
                                cfg = float(pairs[1])
                        case "width:":
                            width = 16 * ((int(pairs[1]) + 8) // 16)
                        case "height:":
                            height = 16 * ((int(pairs[1]) + 8) // 16)
                        case "PAG:":
                            if len(pairs) == 3:
                                PAG_scale = float(pairs[1])
                                PAG_adapt = float(pairs[2].strip('\(\)'))
                        case "LoRA:":
                            if len(pairs) == 3:
                                loraName = pairs[1]
                                loraScale = float(pairs[2].strip('\(\)'))
                        case "Sampler:":
                            if len(pairs) == 3:
                                sampler = f"{pairs[1]} {pairs[2]}"
                            else:
                                sampler = pairs[1]

        return positive, negative, sampler, width, height, seed, steps, cfg, rescale, nr, ng, nb, ns, PAG_scale, PAG_adapt, loraName, loraScale

    resolutionList1024 = [
        (2048, 512),    (1728, 576),    (1408, 704),    (1280, 768),    (1216, 832),
        (1024, 1024),
        (832, 1216),    (768, 1280),    (704, 1408),    (576, 1728),    (512, 2048)
    ]

    def updateWH (dims, w, h):
        #   returns None to dimensions dropdown so that it doesn't show as being set to particular values
        #   width/height could be manually changed, making that display inaccurate and preventing immediate reselection of that option
        #   passing by value because of odd gradio bug? when using index can either update displayed list correctly, or get values correctly, not both
        wh = dims.split('\u00D7')
        return None, int(wh[0]), int(wh[1])

    def toggleSharp ():
        KolorsStorage.sharpNoise ^= True
        return gradio.Button.update(value=['s', 'S'][KolorsStorage.sharpNoise],
                                variant=['secondary', 'primary'][KolorsStorage.sharpNoise])

    def maskFromImage (image):
        if image:
            return image, 'drawn'
        else:
            return None, 'none'


    with gradio.Blocks(analytics_enabled=False, head=canvas_head) as kolors_block:
        with ResizeHandleRow():
            with gradio.Column():
                with gradio.Row():
                    access = ToolButton(value='\U0001F917', variant='secondary', visible=False)
                    CL = ToolButton(value='\u29BE', variant='secondary', tooltip='centre latents to mean')
                    SP = ToolButton(value='ꌗ', variant='secondary', tooltip='prompt enhancement')
                    parse = ToolButton(value="↙️", variant='secondary', tooltip="parse")
                    karras = ToolButton(value="\U0001D542", variant='secondary', tooltip="use Karras sigmas")
                    sampler = gradio.Dropdown(schedulerList, label='Sampler', value="UniPC", type='value', scale=1)

                with gradio.Row():
                    positive_prompt = gradio.Textbox(label='Prompt', placeholder='Enter a prompt here...', lines=2)
                    style = gradio.Dropdown([x[0] for x in styles.styles_list], label='Style', value="(None)", type='index', scale=0)

                with gradio.Row():
                    negative_prompt = gradio.Textbox(label='Negative', lines=1, value="")
                    batch_size = gradio.Number(label='Batch Size', minimum=1, maximum=9, value=1, precision=0, scale=0)
                with gradio.Row():
                    width = gradio.Slider(label='Width', minimum=128, maximum=2048, step=16, value=defaultWidth)
                    height = gradio.Slider(label='Height', minimum=128, maximum=2048, step=16, value=defaultHeight)
                    swapper = ToolButton(value="\U000021C4")
                    dims = gradio.Dropdown([f'{i} \u00D7 {j}' for i,j in resolutionList1024],
                                        label='Quickset', type='value', scale=0)

                with gradio.Row():
                    guidance_scale = gradio.Slider(label='CFG', minimum=1, maximum=16, step=0.1, value=4.0, scale=1)
                    CFGrescale = gradio.Slider(label='rescale CFG', minimum=0.00, maximum=1.0, step=0.01, value=0.0, scale=1)
                    bCFG = ToolButton(value="0", variant='secondary', tooltip="use bias CFG correction")
                with gradio.Row():
                    PAG_scale = gradio.Slider(label='Perturbed-Attention Guidance scale', minimum=0, maximum=8, step=0.1, value=0.0, scale=1)
                    PAG_adapt = gradio.Slider(label='PAG adaptive scale', minimum=0.00, maximum=0.1, step=0.001, value=0.0, scale=1)
                with gradio.Row():
                    steps = gradio.Slider(label='Steps', minimum=1, maximum=80, step=1, value=20, scale=1, visible=True)
                    random = ToolButton(value="\U0001f3b2\ufe0f", variant="primary")
                    sampling_seed = gradio.Number(label='Seed', value=-1, precision=0, scale=0)

                with gradio.Row(equal_height=True):
                    lora = gradio.Dropdown([x for x in loras], label='LoRA (place in models/diffusers/KolorsLora)', value="(None)", type='value', multiselect=False, scale=1)
                    refreshL = ToolButton(value='\U0001f504')
                    scale = gradio.Slider(label='LoRA weight', minimum=-1.0, maximum=1.0, value=1.0, step=0.01, scale=1)

                with gradio.Accordion(label='the colour of noise', open=False):
                    with gradio.Row():
                        initialNoiseR = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='red')
                        initialNoiseG = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='green')
                        initialNoiseB = gradio.Slider(minimum=0, maximum=1.0, value=0.0, step=0.01,  label='blue')
                        initialNoiseA = gradio.Slider(minimum=0, maximum=0.2, value=0.0, step=0.001, label='strength')
                        sharpNoise = ToolButton(value="s", variant='secondary', tooltip='Sharpen initial noise')

                with gradio.Accordion(label='IP Adapter', open=False):
                    with gradio.Row():
                        IPASource = gradio.Image(label='source image', sources=['upload'], type='pil', height=320, show_download_button=False)
                        with gradio.Column():
                            IPAType = gradio.Dropdown(label='Type', choices=["(None)", "Plus"], value="(None)")
                            IPAScale = gradio.Slider(label="Scale", minimum=0.01, maximum=1.0, step=0.01, value=0.5)

                with gradio.Accordion(label='ControlNet', open=False, visible=True):
                    with gradio.Row():
                        CNSource = gradio.Image(label='control image', sources=['upload'], type='pil', height=320, show_download_button=False)
                        with gradio.Column():
                            CNMethod = gradio.Dropdown(['(None)', 'canny', 'depth', 'pose'], 
                                                        label='method', value='(None)', type='index', multiselect=False, scale=1)
                            CNStrength = gradio.Slider(label='Strength', minimum=0.00, maximum=1.0, step=0.01, value=0.8)
                            CNStart = gradio.Slider(label='Start step', minimum=0.00, maximum=1.0, step=0.01, value=0.0)
                            CNEnd = gradio.Slider(label='End step', minimum=0.00, maximum=1.0, step=0.01, value=0.8)

                with gradio.Accordion(label='image to image', open=False):
                    if KolorsStorage.forgeCanvas:
                        i2iSource = ForgeCanvas(elem_id="Kolors_img2img_image", height=320, scribble_color=opts.img2img_inpaint_mask_brush_color, scribble_color_fixed=False, scribble_alpha=100, scribble_alpha_fixed=False, scribble_softness_fixed=False)
                        with gradio.Row():
                            i2iFromGallery = gradio.Button(value='Get gallery image')
                            i2iSetWH = gradio.Button(value='Set size from image')
                            i2iCaption = gradio.Button(value='Caption image')
                            toPrompt = ToolButton(value='P', variant='secondary')
                        
                        with gradio.Row():
                            i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                            AS = ToolButton(value='AS')
                            maskType = gradio.Dropdown(['i2i', 'inpaint mask', 'sketch', 'inpaint sketch'], value='i2i', label='Type', type='index')
                        with gradio.Row():
                            maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=64, step=1, value=0)
                            maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                 
                    else:
                        with gradio.Row():
                            i2iSource = gradio.Image(label='image to image source', sources=['upload'], type='pil', interactive=True, show_download_button=False)
                            if KolorsStorage.usingGradio4:
                                maskSource = gradio.ImageEditor(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, layers=False, brush=gradio.Brush(colors=['#FFFFFF'], color_mode='fixed'))
                            else:
                                maskSource = gradio.Image(label='mask source', sources=['upload'], type='pil', interactive=True, show_download_button=False, tool='sketch', image_mode='RGB', brush_color='#F0F0F0')#opts.img2img_inpaint_mask_brush_color)
                        with gradio.Row():
                            with gradio.Column():
                                with gradio.Row():
                                    i2iDenoise = gradio.Slider(label='Denoise', minimum=0.00, maximum=1.0, step=0.01, value=0.5)
                                    AS = ToolButton(value='AS')
                                with gradio.Row():
                                    i2iFromGallery = gradio.Button(value='Get gallery image')
                                    i2iSetWH = gradio.Button(value='Set size from image')
                                with gradio.Row():
                                    i2iCaption = gradio.Button(value='Caption image (Florence-2)', scale=6)
                                    toPrompt = ToolButton(value='P', variant='secondary')

                            with gradio.Column():
                                maskType = gradio.Dropdown(['none', 'drawn', 'image', 'composite'], value='none', label='Mask', type='index')
                                maskBlur = gradio.Slider(label='Blur mask radius', minimum=0, maximum=25, step=1, value=0)
                                maskCut = gradio.Slider(label='Ignore Mask after step', minimum=0.00, maximum=1.0, step=0.01, value=1.0)
                                maskCopy = gradio.Button(value='use i2i source as template')

                with gradio.Row():
                    noUnload = gradio.Button(value='keep models loaded', variant='primary' if KolorsStorage.noUnload else 'secondary', tooltip='noUnload', scale=1)
                    unloadModels = gradio.Button(value='unload models', tooltip='force unload of models', scale=1)

                if KolorsStorage.forgeCanvas:
                    ctrls = [positive_prompt, negative_prompt, sampler, width, height, guidance_scale, CFGrescale, steps, sampling_seed, batch_size, i2iSource.background, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, i2iSource.foreground, maskBlur, maskCut, IPASource, IPAType, IPAScale, CNSource, CNMethod, CNStrength, CNStart, CNEnd]
                else:
                    ctrls = [positive_prompt, negative_prompt, sampler, width, height, guidance_scale, CFGrescale, steps, sampling_seed, batch_size, i2iSource, i2iDenoise, style, PAG_scale, PAG_adapt, maskType, maskSource, maskBlur, maskCut, IPASource, IPAType, IPAScale, CNSource, CNMethod, CNStrength, CNStart, CNEnd]
                
                parseCtrls = [positive_prompt, negative_prompt, sampler, width, height, sampling_seed, steps, guidance_scale, CFGrescale, initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, PAG_scale, PAG_adapt, lora, scale]

            with gradio.Column():
                generate_button = gradio.Button(value="Generate", variant='primary', visible=True)
                output_gallery = gradio.Gallery(label='Output', height="80vh", type='pil', interactive=False, elem_id="Kolors_gallery",
                                            show_label=False, object_fit='contain', visible=True, columns=3, rows=3, preview=True)

#   caption not displaying linebreaks, alt text does
                gallery_index = gradio.Number(value=0, visible=False)
                infotext = gradio.Textbox(value="", visible=False)
                base_seed = gradio.Number(value=0, visible=False)

                with gradio.Row():
                    buttons = parameters_copypaste.create_buttons(["img2img", "inpaint", "extras"])

                for tabname, button in buttons.items():
                    parameters_copypaste.register_paste_params_button(parameters_copypaste.ParamBinding(
                        paste_button=button, tabname=tabname,
                        source_text_component=infotext,
                        source_image_component=output_gallery,
                    ))


        if KolorsStorage.forgeCanvas:
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource.background, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource.background])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource.background, positive_prompt], outputs=[positive_prompt])
        else:
            maskCopy.click(fn=maskFromImage, inputs=[i2iSource], outputs=[maskSource, maskType])
            i2iSetWH.click (fn=i2iSetDimensions, inputs=[i2iSource, width, height], outputs=[width, height], show_progress=False)
            i2iFromGallery.click (fn=i2iImageFromGallery, inputs=[output_gallery, gallery_index], outputs=[i2iSource])
            i2iCaption.click (fn=i2iMakeCaptions, inputs=[i2iSource, positive_prompt], outputs=[positive_prompt])

        noUnload.click(toggleNU, inputs=None, outputs=noUnload)
        unloadModels.click(unloadM, inputs=None, outputs=None, show_progress=True)

        SP.click(toggleSP, inputs=None, outputs=SP).then(superPrompt, inputs=[positive_prompt, sampling_seed], outputs=[SP, positive_prompt])
        karras.click(toggleKarras, inputs=None, outputs=karras)
        sharpNoise.click(toggleSharp, inputs=None, outputs=sharpNoise)
        dims.input(updateWH, inputs=[dims, width, height], outputs=[dims, width, height], show_progress=False)
        parse.click(parsePrompt, inputs=parseCtrls, outputs=parseCtrls, show_progress=False)
        access.click(toggleAccess, inputs=None, outputs=access)
        bCFG.click(toggleBiasCFG, inputs=None, outputs=bCFG)
        swapper.click(lambda w, h: (h, w), inputs=[width, height], outputs=[width, height], show_progress=False)
        random.click(toggleRandom, inputs=None, outputs=random, show_progress=False)
        AS.click(toggleAS, inputs=None, outputs=AS)
        CL.click(toggleCL, inputs=None, outputs=CL)
        refreshL.click(refreshLoRAs, inputs=None, outputs=[lora])

        toPrompt.click(toggleC2P, inputs=None, outputs=[toPrompt])

        output_gallery.select(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index, show_progress=False).then(fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed], show_progress=False)

        generate_button.click(toggleGenerate, inputs=[initialNoiseR, initialNoiseG, initialNoiseB, initialNoiseA, lora, scale], outputs=[generate_button, SP]).then(predict, inputs=ctrls, outputs=[base_seed, SP, output_gallery], show_progress='full').then(fn=afterGenerate, inputs=None, outputs=generate_button).then(fn=getGalleryIndex, js="selected_gallery_index", inputs=gallery_index, outputs=gallery_index, show_progress=False).then(fn=getGalleryText, inputs=[output_gallery, gallery_index, base_seed], outputs=[infotext, sampling_seed], show_progress=False)

    return [(kolors_block, "Kolors", "kolors_DoE")]

script_callbacks.on_ui_tabs(on_ui_tabs)

