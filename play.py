import torch as tc


@tc.no_grad()
def play(env, agent, comm, args):
    if comm.Get_rank() == 0:
        t = 0
        total_reward = 0.0
        o_t = env.reset()
        while t < args.max_timesteps:
            _ = env.render()
            pi_dist, vpred, _ = agent(tc.tensor(o_t).float().unsqueeze(0))
            a_t = pi_dist.sample()
            o_tp1, r_t, done_t, _ = env.step(a_t.squeeze(0).detach().numpy())
            total_reward += r_t
            t += 1
            if done_t:
                print(f"Episode finished after {t} timesteps.")
                break
            o_t = o_tp1

        print(f"Total reward was {total_reward}.")
