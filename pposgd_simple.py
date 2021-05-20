"""
Ported from
https://github.com/openai/baselines/blob/master/baselines/ppo1/pposgd_simple.py
"""

import numpy as np
import time
from collections import deque
import torch as tc
from mpi4py import MPI
import os

from common.dataset import Dataset
from common.math_util import explained_variance
from common.console_util import fmt_row
from common.misc_util import zipsame
from common.mpi_moments import mpi_moments
import logger


@tc.no_grad()
def traj_segment_generator(agent, env, horizon):
    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    logprobs = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    agent.eval()
    while True:
        prevac = ac
        pi_dist, vpred = agent(tc.tensor(ob).float().unsqueeze(0))

        ac = pi_dist.sample()
        logprob = pi_dist.log_prob(ac)

        ac = ac.squeeze(0).detach().numpy()
        logprob = logprob.squeeze(0).detach().numpy()
        vpred = vpred.squeeze(0).detach().numpy()

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value.
        #
        # We return it as nextvpred, then log everything for timestep T later,
        # after the yield.
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "logprobs": logprobs, "vpred" : vpreds,
                   "new" : news, "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []

        i = t % horizon
        obs[i] = ob
        logprobs[i] = logprob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def compute_losses(batch, agent, entcoeff, clip_param):
    # get relevant info from minibatch dict
    mb_obs = batch["ob"]
    mb_acs = batch["ac"]
    mb_logpi_old = batch["logprobs"]
    mb_advs = batch["adv"]
    mb_vtargs = batch["vtarg"]

    # cast to correct type
    mb_obs = tc.tensor(mb_obs).float().detach()
    mb_acs = tc.tensor(mb_acs).long().detach()
    mb_logpi_old = tc.tensor(mb_logpi_old).float().detach()
    mb_advs = tc.tensor(mb_advs).float().detach()
    mb_vtargs = tc.tensor(mb_vtargs).float().detach()

    # evaluate observations using agent
    mb_pi_dist, mb_vpred_new = agent(mb_obs)
    mb_logpi_new = mb_pi_dist.log_prob(mb_acs)

    # entropy
    ent = mb_pi_dist.entropy()
    meanent = tc.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    # ppo policy loss
    policy_ratio = tc.exp(mb_logpi_new - mb_logpi_old)
    clipped_policy_ratio = tc.clip(policy_ratio, 1.0 - clip_param, 1.0 + clip_param)
    surr1 = mb_advs * policy_ratio
    surr2 = mb_advs * clipped_policy_ratio
    pol_surr = -tc.mean(tc.min(surr1, surr2))

    # ppo value loss
    vf_loss = tc.mean(tc.square(mb_vtargs - mb_vpred_new))

    return pol_surr, pol_entpen, vf_loss, meanent


def learn(env, agent, optimizer, scheduler, comm,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_batchsize, # optimization hypers
        gamma, lam, # advantage estimation
        checkpoint_dir, model_name,
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0, schedule='linear'
    ):

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(agent, env, timesteps_per_actorbatch)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    gradient_steps_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "ent"]

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        logger.log("********** Iteration %i ************"%iters_so_far)

        epsilon_mult_dict = {
            'constant': 1.0,
            'linear': max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        }
        current_clip_param = epsilon_mult_dict[schedule] * clip_param

        seg = next(seg_gen)
        add_vtarg_and_adv(seg, gamma, lam)

        ob, ac, logprobs, adv, tdlamret = seg["ob"], seg["ac"], seg["logprobs"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"]  # predicted value function before udpate
        adv = (adv - adv.mean()) / adv.std()  # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, logprobs=logprobs, adv=adv, vtarg=tdlamret), deterministic=False) # nonrecurrent

        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        agent.train()
        for _ in range(optim_epochs):
            losses = []  # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                pol_surr, pol_entpen, vf_loss, ent = compute_losses(batch, agent, entcoeff, current_clip_param)
                total_loss = pol_surr + pol_entpen + vf_loss

                optimizer.zero_grad()
                total_loss.backward()
                with tc.no_grad():
                    for p in agent.parameters():
                        g_old = p.grad.numpy()
                        g_new = np.zeros_like(g_old)
                        comm.Allreduce(sendbuf=g_old, recvbuf=g_new, op=MPI.SUM)
                        p.grad.copy_(tc.tensor(g_new).float() / comm.Get_size())

                optimizer.step()
                scheduler.step()
                gradient_steps_so_far += 1

                # sync agent parameters from process with rank zero. should stay synced automatically,
                # this is just a failsafe
                if gradient_steps_so_far > 0 and gradient_steps_so_far % 100 == 0:
                    with tc.no_grad():
                        for p in agent.parameters():
                            p_data = p.data.numpy()
                            comm.Bcast(p_data, root=0)
                            p.data.copy_(tc.tensor(p_data).float())

                newlosses = (
                    pol_surr.detach().numpy(),
                    pol_entpen.detach().numpy(),
                    vf_loss.detach().numpy(),
                    ent.detach().numpy()
                )
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch, agent, entcoeff, current_clip_param)
            losses.append(tuple(list(map(lambda loss: loss.detach().numpy(), list(newlosses)))))
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_" + name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"])  # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal)  # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if comm.Get_rank() == 0:
            logger.dump_tabular()
            if iters_so_far > 0 and iters_so_far % 10 == 0:
                print("Saving checkpoint...")
                os.makedirs(os.path.join(checkpoint_dir, model_name), exist_ok=True)
                tc.save(agent.state_dict(), os.path.join(checkpoint_dir, model_name, 'model.pth'))


def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
