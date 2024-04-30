import torch
import os
import os.path as osp
from jutils import image_utils
import numpy as np
import kiui
import pickle
from cores.lib.guidance import StableDiffusion

device = 'cuda:0'
save_path = 'results/debug'
name = 'hehe'
i = 0

torch.manual_seed(123)
def get_fake_data():

    N = 2
    F = 4
    latents = torch.ones([1, 4, 32, 32], device=device).repeat(N, 1, 1, 1)
    t = torch.tensor(958).to(device)
    text_embeddings = torch.ones([N, 77, 1024], device=device)
    camera = torch.ones([F, 16], device=device)
    
    return latents, t, text_embeddings, camera


def load():
    with open('results/guidance.pkl', 'rb') as f:
        guidance = pickle.load(f)
    assert isinstance(guidance, StableDiffusion)
    with open('results/data.pkl', 'rb') as f:
        save = pickle.load(f)
        data = save['data']
        cfg = save['cfg']
    pipe = guidance.pipe
    return guidance, pipe, data, cfg


def run():
    guidance, pipe, data, cfg = load()
    # latents, t, text_embeddings, camera = get_fake_data() 
    # out_guid = guidance.forward_mv_unet(latents, t, text_embeddings, camera, get_inp=True)
    # out_pipe_inp = pipe.debug_call(
    #         negative_prompt="",
    #         guidance_scale=5,
    #         num_inference_steps=30,
    #         camera=camera,
    #         num_frames=4,
    #         # latents=latents[0:1].unsqueeze(1).repeat(1, 4, 1, 1, 1).reshape(-1, 4, 32, 32),
    #         text_embedding=text_embeddings)
    # for key in out_guid:
    #     if torch.is_tensor(out_guid[key]):
    #         if  torch.any(torch.not_equal(out_guid[key], out_pipe_inp[key])):
    #             print(key, )
    #             print('guidance', out_guid[key])
    #             print('pipe', out_pipe_inp[key])
    #             print('---')
    #     else:
    #         print('int', key, out_guid[key])
    #         print('===')

    # # output
    # pipe_out = pipe.unet(**out_pipe_inp)
    # guidance_out = guidance.forward_mv_unet(latents, t, text_embeddings, camera)
    # print(pipe_out[:2] - guidance_out)
    # print(torch.any(torch.not_equal(pipe_out[:2], guidance_out)))


    def run_full(T=20):
        for i in range(1):
            latents = torch.randn([4, 4, 32, 32], device=device)
            guidance.scheduler.set_timesteps(T)
            _, images, rtn = guidance.prompt_to_img([cfg.guidance.text],
                                    [""], 
                                    guidance.res, guidance.res,
                                    # num_inference_steps=50,
                                    num_inference_steps=T,

                                    guidance_scale=5,
                                    camera=data['camera'],
                                    latents=latents,
                                    ) # (4, 16)

            print(cfg.guidance.text)
            print(osp.join(save_path, f'{name}_{i:04d}_prompt'))
            image_utils.save_images(images, osp.join(save_path, f'{name}_{i:04d}_prompt'))

            pipe.scheduler.set_timesteps(T)
            print(rtn['latents'].shape)
            image, rtn = pipe(
                cfg.guidance.text,
                negative_prompt="",
                guidance_scale=5,
                num_inference_steps=T,
                camera=data['camera'],
                num_frames=4,
                text_embedding=rtn['text_embeds'],
                latents=None, # rtn['latents'].repeat(4, 1, 1, 1),
            )
            grid = np.concatenate(image, axis=1)
            kiui.write_image(osp.join(save_path, f'{name}_{i:04d}_pipeline.jpg'), grid)
    
            print(rtn['latents'].shape)
            latents = rtn['latents'] # .reshape(-1, 4, 4, 32, 32)[:, 0]
            _, images, rtn = guidance.prompt_to_img([cfg.guidance.text],
                                    [""], 
                                    guidance.res, guidance.res,
                                    # num_inference_steps=50,
                                    num_inference_steps=T,
                                    guidance_scale=5,
                                    camera=data['camera'],
                                    latents=latents,
                                    ) # (4, 16)

            print(cfg.guidance.text)
            image_utils.save_images(images, osp.join(save_path, f'{name}_{i:04d}_prompt2'))

    run_full()
        

def script():
    ones = torch.ones([1, 4, 32, 32], device=device)
    twos = ones * 2

    inp = torch.cat([ones, twos])
    a, b = torch.chunk(inp, 2)
    print(a, b)


if __name__ == "__main__":
    run() 
    # script()
