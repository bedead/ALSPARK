import copy
import random
import torch
from torch import autocast
from pytorch_lightning import seed_everything
import cv2
import numpy as np
from basicsr.utils import tensor2img
from ldm.inference_base import (
    diffusion_inference,
    get_adapters,
    get_sd_models,
)
from ldm.modules.extra_condition import api
from ldm.modules.extra_condition.api import ExtraCondition, get_cond_model
from ldm.modules.encoders.adapter import CoAdapterFuser


class StyleTransferPipeline:
    def __init__(self, global_opt):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.global_opt = global_opt
        self.sd_model, self.sampler = get_sd_models(global_opt)
        self.adapters = {}
        self.cond_models = {}
        self.coadapter_fuser = CoAdapterFuser(
            unet_channels=[320, 640, 1280, 1280], width=768, num_head=8, n_layes=3
        )
        self.coadapter_fuser.load_state_dict(
            torch.load("models/coadapter-fuser-sd15v1.pth")
        )
        self.coadapter_fuser = self.coadapter_fuser.to(global_opt.device)
        torch.cuda.empty_cache()

    def preprocess_images(self, im1_canny, im1_style):
        """Resize and preprocess input images to match dimensions."""
        h, w = None, None
        if im1_canny is not None:
            h, w, _ = im1_canny.shape
        elif im1_style is not None:
            h, w, _ = im1_style.shape

        if h and w:
            if im1_canny is not None:
                im1_canny = cv2.resize(im1_canny, (w, h), interpolation=cv2.INTER_CUBIC)
            if im1_style is not None:
                im1_style = cv2.resize(im1_style, (w, h), interpolation=cv2.INTER_CUBIC)

        return im1_canny, im1_style

    def prepare_conditions(self, opt, input_images, cond_weights, cond_names):
        """Prepare condition inputs for the model."""
        conds = []
        activated_conds = []

        for idx, (input_image, cond_weight, cond_name) in enumerate(
            zip(input_images, cond_weights, cond_names)
        ):
            if input_image is not None:
                activated_conds.append(cond_name)
                if cond_name in self.adapters:
                    self.adapters[cond_name]["model"] = self.adapters[cond_name][
                        "model"
                    ].to(self.global_opt.device)
                else:
                    self.adapters[cond_name] = get_adapters(
                        opt, getattr(ExtraCondition, cond_name)
                    )
                self.adapters[cond_name]["cond_weight"] = cond_weight

                process_cond_module = getattr(api, f"get_cond_{cond_name}")
                if cond_name not in self.cond_models:
                    self.cond_models[cond_name] = get_cond_model(
                        opt, getattr(ExtraCondition, cond_name)
                    )
                conds.append(
                    process_cond_module(
                        opt,
                        input_image,
                        "image",
                        self.cond_models[cond_name],
                    )
                )
            else:
                if cond_name in self.adapters:
                    self.adapters[cond_name]["model"] = self.adapters[cond_name][
                        "model"
                    ].cpu()

        return conds, activated_conds

    def run(
        self,
        im1_canny,
        im1_style,
        cond_weight_style,
        cond_weight_canny,
        prompt,
        neg_prompt,
        scale,
        n_samples,
        steps,
        resize_short_edge,
        cond_tau,
    ):
        """Main pipeline to generate stylized images."""
        with torch.inference_mode(), self.sd_model.ema_scope(), autocast(self.device):
            # Setup options
            opt = copy.deepcopy(self.global_opt)
            opt.seed = random.randint(1, 4000000000)  # Random seed
            (
                opt.prompt,
                opt.neg_prompt,
                opt.scale,
                opt.n_samples,
                opt.steps,
                opt.resize_short_edge,
                opt.cond_tau,
            ) = (
                prompt,
                neg_prompt,
                scale,
                n_samples,
                steps,
                resize_short_edge,
                cond_tau,
            )

            if im1_canny is not None:
                im1_canny = np.array(im1_canny)
            if im1_style is not None:
                im1_style = np.array(im1_style)

            # Preprocess input images
            im1_canny, im1_style = self.preprocess_images(im1_canny, im1_style)
            input_images = [im1_canny, im1_style]
            cond_weights = [cond_weight_canny, cond_weight_style]
            cond_names = ["canny", "style"]

            # Prepare conditions
            conds, activated_conds = self.prepare_conditions(
                opt, input_images, cond_weights, cond_names
            )

            # Generate features
            features = dict()
            for idx, cond_name in enumerate(activated_conds):
                cur_feats = self.adapters[cond_name]["model"](conds[idx])
                if isinstance(cur_feats, list):
                    for i in range(len(cur_feats)):
                        cur_feats[i] *= self.adapters[cond_name]["cond_weight"]
                else:
                    cur_feats *= self.adapters[cond_name]["cond_weight"]
                features[cond_name] = cur_feats

            # Fuse features
            adapter_features, append_to_context = self.coadapter_fuser(features)

            # Generate output images
            ims = []
            seed_everything(opt.seed)
            for _ in range(opt.n_samples):
                result = diffusion_inference(
                    opt,
                    self.sd_model,
                    self.sampler,
                    adapter_features,
                    append_to_context,
                )
                ims.append(tensor2img(result, rgb2bgr=False))

            torch.cuda.empty_cache()
            return ims[0]  # Return the first image


# Initialize pipeline
def create_pipeline(global_opt):
    return StyleTransferPipeline(global_opt)
