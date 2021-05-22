import torch as tc
import moviepy.editor as mpy
import numpy as np
import uuid
import os
from collections import deque
import pyglet
import matplotlib.pyplot as plt
import copy
import cv2


@tc.no_grad()
def movie(env, agent, args):
    base_path = os.path.join(args.asset_dir, args.model_name)
    os.makedirs(base_path, exist_ok=True)

    fps = 64
    max_frames = 2048
    queue = deque(maxlen=max_frames)

    def make_video():
        def make_frame(t):
            # t will range from 0 to (self.max_frames / self.fps).
            frac_done = t / (max_frames // fps)
            max_idx = len(queue) - 1
            idx = int(max_idx * frac_done)
            arr_fp = queue[idx]
            x = plt.imread(arr_fp)
            return (255 * x).astype(np.int32).astype(np.uint8)

        filename = f"{uuid.uuid4()}.gif"
        fp = os.path.join(base_path, filename)

        clip = mpy.VideoClip(make_frame, duration=(max_frames // fps))
        clip.write_gif(fp, fps=fps)

        print(f"Saving video to {fp}")

    t = 0
    total_reward = 0.0
    o_t = env.reset()
    while t < args.env_steps:
        pi_dist, vpred = agent(tc.tensor(o_t).float().unsqueeze(0))
        a_t = pi_dist.sample()
        o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy())

        arr = env.render(mode='rgb_array')
        arr_fp = f"/tmp/{str(uuid.uuid4())}.png"
        plt.imsave(arr=arr, fname=arr_fp)
        queue.append(arr_fp)

        total_reward += r_t
        t += 1
        if done_t:
            make_video()
            break
        o_t = o_tp1
