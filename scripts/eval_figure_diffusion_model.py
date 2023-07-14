import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        nargs="?",
        help="Checkpoint path (file path)",
        default="/mnt/colab_public/datasets/joan/LatentDiffusion/models/ldm/9f89f99b16ed220aacff46ea08450e48_paper2fig-snowBert-kl-f8-8layers/checkpoints/epoch=000299.ckpt"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=16,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    opt = parser.parse_args()

    # Compute ckpt path and config path

    # if file has extension ckpt
    if os.path.isfile(opt.ckpt):
        ckpt_path = opt.ckpt
        config_path = glob.glob(os.path.join(os.path.dirname(os.path.dirname(opt.ckpt)), "configs/*project.yaml"))[0]
        dirname = f"evaluation_cfg_{str(opt.scale)}"
        out_dir = os.path.join(os.path.dirname(os.path.dirname(opt.ckpt)), dirname) 
    else:
        # error
        raise ValueError("Invalid ckpt path")

    config = OmegaConf.load(config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, ckpt_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(out_dir, exist_ok=True)
    outpath = out_dir

    # define validation dataset
    dataset = instantiate_from_config(config.data.params.test)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        with model.ema_scope():
            batch_id = 0
            for batch in tqdm(dataloader):
                sample_path = os.path.join(outpath, f"sample_{str(batch_id)}")
                base_count =  len(glob.glob(os.path.join(sample_path, "*.png"))) 
                if base_count > 0:
                    print(f"Skipping {sample_path}")
                    batch_id += 1
                    continue
                os.makedirs(sample_path, exist_ok=True)
                batch_id += 1
                caption = batch['caption'][0]
                with open(os.path.join(sample_path, "caption.txt"), "w") as f:
                    f.write(caption)
                for n in trange(opt.n_iter, desc="Sampling"):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(opt.n_samples * [""])
                        c = model.get_learned_conditioning(opt.n_samples * [caption])
                        shape = [4, opt.H//8, opt.W//8]
                        samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                        for x_sample in x_samples_ddim:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                            base_count += 1

    print("Done")
