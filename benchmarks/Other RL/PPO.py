# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:07:24 2026

@author: MR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import time
from pathlib import Path


# ===============================
# Preferences Settings
# ===============================
np.set_printoptions(suppress=True)
pd.set_option('display.max_columns', None)
plt.rcParams['figure.dpi'] = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================
# Networks
# ===============================

class RequestView(nn.Module):
    """ Encode request features. """
    def __init__(self, in_dim=14, hidden_dim=64, out_dim=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim))
    
    def forward(self, x):
        return self.net(x)

class VehicleView(nn.Module):
    """ Encode vehicle trajectory features. """
    def __init__(self, in_dim=3, hidden_dim=64, out_dim=64, drop=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=in_dim, hidden_size=hidden_dim,
                            batch_first=True, bidirectional=False)
        
        self.out_head = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(drop),
                    nn.Linear(hidden_dim, out_dim))
    
    def forward(self, padded, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths.cpu(),
            batch_first=True, enforce_sorted=False)
        
        _, (h,_) = self.lstm(packed)
        return self.out_head(h[-1])

class ContextView(nn.Module):
    """ Encode request-vehicle contextual features. """
    def __init__(self, in_dim=5, hidden_dim=64, out_dim=64, drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden_dim, out_dim))
    
    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):
    """Action: Sampling vehicle selection + ETA"""
    def __init__(self, in_dim=192, hidden1_dim=64, hidden2_dim=64,
                 drop1=0.1, drop2=0.1):
        super().__init__()
        
        self.samp_vehi_net = nn.Sequential(
            nn.Linear(in_dim, hidden1_dim),
            nn.ReLU(),
            nn.Dropout(drop1),
            nn.Linear(hidden1_dim, 1))
        
        self.samp_eta_net = nn.Sequential(
            nn.Linear(in_dim, hidden2_dim),
            nn.ReLU(),
            nn.Dropout(drop2),
            nn.Linear(hidden2_dim, 3))

    def forward(self, fusion_x):
        car_logits = self.samp_vehi_net(fusion_x).squeeze(-1)  # (K+1,)
        eta_logits = self.samp_eta_net(fusion_x)                       # (K+1, 3)
        return car_logits, eta_logits
    

# ===============================
# Instance
# ===============================
'''initialize data'''
class Instance:
    def __init__(self, path, top_n=10):
        data = pd.read_parquet(path)
        # standardize time
        t_base = data['pick_early'].min()
        time_cols = ['pick_early', 'pick_late', 'drop_early', 'drop_late']
        data[time_cols] = (data[time_cols] - t_base).apply(
            lambda col: col.dt.total_seconds() / 3600)
        
        # add request's ID and status
        data['id'] = np.arange(len(data), dtype=int)
        data['status'] = True

        self.data = data.copy()

# ===============================
# Reinforcement Learining
# ===============================
'''RL-agent'''
class Learning(nn.Module):
    def __init__(self, 
                 max_episode=1000, γ=0.5, reward_scale=0.01, 
                 τ=1.0, τ_max=10.0, τ_min=0.0, switch_cap=10, off_cap=50, 
                 reject_loss=-5, 
                 lr_actor=1e-4,
                 ppo_clip=0.2, ppo_epochs=4, ppo_batch_size=64,
                 entropy_coef=0.0, normalize_advantage=True):
        super().__init__()
        # episodes
        self.episode = 0
        self.max_episode = max_episode
        self.epoch = 0
        
        # ARS module
        self.ARS = True
        self.off_cap = off_cap # episode when ARS turns off
        
        self.τ = τ
        self.τ_max = τ_max
        self.τ_min = τ_min
        self.heat_coef = 0.5
        self.heat_gate = True  # temperature gate (switch heat/cool)
        self.fail = 0          # global non-improvement
        self.temp_fail = 0     # non-improvement under extreme temperature
        self.switch_cap = switch_cap   # how many fails switch the temperature gate
        
        # networks
        self.request_view = RequestView().to(device)
        self.vehicle_view = VehicleView().to(device)
        self.context_view = ContextView().to(device)
        self.actor = Actor().to(device)
        self.dummy_vehicle_x = nn.Parameter(torch.zeros(1, 64, device=device))
        
        # optimizers
        self.actor_params = (list(self.request_view.parameters())
                             + list(self.vehicle_view.parameters())
                             + list(self.context_view.parameters())
                             + list(self.actor.parameters())
                             + [self.dummy_vehicle_x])
        self.actor_opti = torch.optim.Adam(self.actor_params, lr=lr_actor)
        
        # auximilary parameters
        self.ov_epi = np.full(max_episode, np.nan)
        self.fail = 0           # times of non-improvement
        self.overtemp_fail = 0  # times of non-improvement on temperature bound
        
        # hyper-parameters
        self.γ = γ
        self.β = reward_scale
        self.λ = reject_loss
        self.ppo_clip = ppo_clip
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.entropy_coef = entropy_coef
        self.normalize_advantage = normalize_advantage
        self.memory = []
        
        # records
        self.ov_best = -np.inf
        self.returns = np.full(self.max_episode, np.nan)
        self.recent_best = -np.inf; self.recent = 20
        self.runtime = 0
        
    def reward_curve(self):
        plt.plot(np.arange(self.max_episode), 
                 self.episode_ov, color='tab:red')
        plt.show()
        
    def act(self, req, avail_vehs, debug_plot=False, greedy_sample=False):
        """ Two-step action:
            1. select vehicle k in {0, ..., K-1} or dummy K;
            2. if real vehicle is selected, select ETA in {0, 1, 2}.
        """
        # value current state value
        avail_vehs = list(avail_vehs)

        # featurize
        req_ftrs = self.featurize_request(req)
        vehi_padded, vehi_lengths = self.featurize_vehicle_route(avail_vehs,
                                                                 from_head=True)
        context_ftrs = self.featurize_context(req, avail_vehs)
        
        # request view: (1, 64)
        req_x = torch.tensor(req_ftrs, dtype=torch.float32,
                             device=device).unsqueeze(0)
        req_x = self.request_view(req_x)
        
        # vehicle view: route suffixes from heads + dummy vehicle, (K+1, 64)
        real_vehi_x = self.vehicle_view(vehi_padded, vehi_lengths)
        vehi_x = torch.cat([real_vehi_x, self.dummy_vehicle_x], dim=0)
        
        # context view: (K+1, 64)
        context_x = torch.tensor(context_ftrs, dtype=torch.float32, device=device)
        context_x = self.context_view(context_x)
 
        # fusion features: (K+1, 192)
        fusion_x = torch.cat([
            req_x.expand(vehi_x.size(0), -1),
            vehi_x,
            context_x], dim=1)
        
        # to actor
        car_logits, eta_logits = self.actor(fusion_x)
        # sample vehicle (including dummy)
        π_car = torch.distributions.Categorical(logits=car_logits)
        car_pos = torch.argmax(car_logits) if greedy_sample else π_car.sample()

        car_idx = int(car_pos.item())
        log_prob = π_car.log_prob(car_pos)

        transition = {
            'avail_vehs': avail_vehs,
            'req_ftrs': req_ftrs,
            'vehi_padded': vehi_padded.detach().cpu(),
            'vehi_lengths': vehi_lengths.detach().cpu(),
            'context_ftrs': context_ftrs,
            'car_pos': car_idx,
            'eta_pos': None,
        }

        # dummy vehicle: reject / no dispatch for now
        if car_idx == len(avail_vehs):
            transition['old_log_prob'] = float(log_prob.detach().cpu())
            return car_idx, None, transition

        # sample ETA under the selected real vehicle
        π_eta = torch.distributions.Categorical(logits=eta_logits[car_idx])
        eta = torch.argmax(eta_logits[car_idx]) if greedy_sample else π_eta.sample()
        log_prob = log_prob + π_eta.log_prob(eta)

        if debug_plot and self.epoch % 500 == 0:
            logits_np = car_logits.detach().cpu().numpy()
            probs_np = π_car.probs.detach().cpu().numpy()
            plt.bar(np.arange(len(logits_np)), logits_np)
            plt.title('Vehicle logits')
            plt.show()
            plt.close()
            plt.bar(np.arange(len(probs_np)), probs_np)
            plt.ylim(0, 1.1)
            plt.title('Vehicle probabilities')
            plt.show()
            plt.close()
            
        transition['eta_pos'] = int(eta.item())
        transition['old_log_prob'] = float(log_prob.detach().cpu())
        return int(avail_vehs[car_idx]), int(eta.item()), transition

    def reward(self, Δz, is_dummy):
        """Training reward based ARS"""
        # -10 propability callpase
        raw = self.λ * self.τ if is_dummy else float(Δz)
        reward = torch.tensor(raw, dtype=torch.float32, device=device)
        return torch.tanh(self.β * reward)

    def _policy_logits(self, req_ftrs, vehi_padded, vehi_lengths, context_ftrs):
        req_x = torch.as_tensor(req_ftrs, dtype=torch.float32,
                                device=device).unsqueeze(0)
        req_x = self.request_view(req_x)

        vehi_padded = vehi_padded.to(device)
        vehi_lengths = vehi_lengths.to(device)
        real_vehi_x = self.vehicle_view(vehi_padded, vehi_lengths)
        vehi_x = torch.cat([real_vehi_x, self.dummy_vehicle_x], dim=0)

        context_x = torch.as_tensor(context_ftrs, dtype=torch.float32,
                                    device=device)
        context_x = self.context_view(context_x)

        fusion_x = torch.cat([
            req_x.expand(vehi_x.size(0), -1),
            vehi_x,
            context_x], dim=1)
        return self.actor(fusion_x)

    def _transition_log_prob_entropy(self, transition):
        car_logits, eta_logits = self._policy_logits(
            transition['req_ftrs'],
            transition['vehi_padded'],
            transition['vehi_lengths'],
            transition['context_ftrs'])

        car_dist = torch.distributions.Categorical(logits=car_logits)
        car_pos = torch.tensor(transition['car_pos'], dtype=torch.long,
                               device=device)
        log_prob = car_dist.log_prob(car_pos)
        entropy = car_dist.entropy()

        eta_pos = transition['eta_pos']
        if eta_pos is not None:
            eta_dist = torch.distributions.Categorical(
                logits=eta_logits[transition['car_pos']])
            eta_pos = torch.tensor(eta_pos, dtype=torch.long, device=device)
            log_prob = log_prob + eta_dist.log_prob(eta_pos)
            entropy = entropy + eta_dist.entropy()

        return log_prob, entropy

    def update(self, transition, delta_ov, is_dummy, show=False):
        '''Collect one transition; PPO updates on mini rollouts.'''
        reward = self.reward(delta_ov, is_dummy)
        transition['reward'] = float(reward.detach().cpu())
        self.memory.append(transition)

        if len(self.memory) >= self.ppo_batch_size:
            self.ppo_update(show=show)

    def ppo_update(self, show=False):
        if len(self.memory) == 0:
            return

        rewards = torch.tensor([item['reward'] for item in self.memory],
                               dtype=torch.float32, device=device)
        advantages = rewards
        if self.normalize_advantage and len(self.memory) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8)

        old_log_probs = torch.tensor(
            [item['old_log_prob'] for item in self.memory],
            dtype=torch.float32, device=device)

        last_loss = None
        for _ in range(self.ppo_epochs):
            losses = []
            entropies = []
            for idx, transition in enumerate(self.memory):
                log_prob, entropy = self._transition_log_prob_entropy(transition)
                ratio = torch.exp(log_prob - old_log_probs[idx])
                clipped_ratio = torch.clamp(
                    ratio, 1.0 - self.ppo_clip, 1.0 + self.ppo_clip)
                surrogate = torch.min(
                    ratio * advantages[idx],
                    clipped_ratio * advantages[idx])
                losses.append(-surrogate)
                entropies.append(entropy)

            actor_loss = torch.stack(losses).mean()
            entropy_bonus = torch.stack(entropies).mean()
            loss = actor_loss - self.entropy_coef * entropy_bonus

            self.actor_opti.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_params, max_norm=1.0)
            self.actor_opti.step()
            self.epoch += 1
            last_loss = float(loss.detach().cpu())

        if show:
            print(f'Epoch: {self.epoch}')
            print(f'PPO batch size: {len(self.memory)} | loss: {last_loss:.4f}')

        self.memory.clear()

    def featurize_route(self, from_head=False):
        ftrs = []
        lengths = []
        
        for k in range(K):
            route = RH.sol[k]
            head = RH.heads[k] if from_head else 0
            
            # embed distance and temporal trajactory
            time_arr = (route[head:,2] - RH.t0) / (RH.T - RH.t0)
            dis_arr = route[head:,3]
            traj = np.stack([time_arr, dis_arr], axis=1).astype(np.float32)
            traj = torch.as_tensor(traj, dtype=torch.float32, device=device)
            
            ftrs.append(traj)
            lengths.append(traj.size(0))
        
        padded = nn.utils.rnn.pad_sequence(ftrs, batch_first=True)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        
        return padded, lengths
 

    def featurize_request(self, req):
        req_fare = req['fare']

        req_time = req['pick_time'] + (req['drop_time'] - req['pick_time']) / 2
        weekday = req_time.weekday()
        daytime = req_time.hour + req_time.minute / 60 + req_time.second / 3600

        wd_cos = np.cos(2*np.pi * weekday / 7)
        wd_sin = np.sin(2*np.pi * weekday / 7)
        dt_cos = np.cos(2*np.pi * daytime / 24)
        dt_sin = np.sin(2*np.pi * daytime / 24)

        pick_longt = (req['pick_longt'] - longt0) * np.cos(np.radians(lat0))
        pick_lat = req['pick_lat'] - lat0
        drop_longt = (req['drop_longt'] - longt0) * np.cos(np.radians(lat0))
        drop_lat = req['drop_lat'] - lat0

        dx = drop_longt - pick_longt
        dy = drop_lat - pick_lat
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        bearing_cos = dx / norm
        bearing_sin = dy / norm

        expire_slack = req['pick_late'] - RH.now
        drop_early = req['drop_early'] - RH.now
        drop_late = req['drop_late'] - RH.now
        return np.array([
            req_fare / 10,   # scale
            wd_cos, wd_sin, dt_cos, dt_sin,
            pick_longt, pick_lat, drop_longt, drop_lat,
            bearing_cos, bearing_sin,
            expire_slack, drop_early, drop_late], dtype=np.float32)

    def featurize_context(self, req, vehs):
        """
        Build per-vehicle terminal context.
        Context view drops pick_early but keeps fare:
            [fare, dis(k,o), pick_late-tail,
             drop_early-tail, drop_late-tail]
        """
            
        pick_coord = np.array([req['pick_longt'], req['pick_lat']])
        ftrs = []

        for k in vehs:
            tail_node = int(RH.sol[k][-1, 0])
            tail_time = RH.sol[k][-1, 2]
            tail_coord = RH.get_node_coord(tail_node)
            
            dis_topick = RH.get_distance(tail_coord, pick_coord)  # distance to pickup

            ftrs.append([
                float(req['fare']) / 10,
                float(dis_topick),
                float(req['pick_late'] - tail_time),
                float(req['drop_early'] - tail_time),
                float(req['drop_late'] - tail_time)])

        dummy_ftr = [float(req['fare']) / 10, 9.9, 0.0, 0.0, 0.0]
        ftrs.append(dummy_ftr)
        return np.asarray(ftrs, dtype=np.float32)

    def featurize_vehicle_route(self, vehs, from_head=True):
        """Build padded route tensors for actor vehicle view."""
        ftrs = []
        lengths = []
        horizon = max(RH.T - RH.t0, 1e-6)
        
        for k in vehs:
            route = RH.sol[k]
            head = int(RH.heads[k]) if from_head else 0
            head = min(max(head, 0), len(route) - 1)
            seq = []
            
            # embed future route suffix from the available head
            for row in route[head:]:
                node_id = int(row[0])
                coord = RH.get_node_coord(node_id)
                xy = RH.normalize_coord(coord)
                time_norm = (float(row[2]) - RH.t0) / horizon
                seq.append([xy[0], xy[1], time_norm])
            
            if len(seq) == 0:
                node_id = int(route[-1,0])
                coord = RH.get_node_coord(node_id)
                xy = RH.normalize_coord(coord)
                seq.append([xy[0], xy[1], 0.0])
            
            traj = torch.as_tensor(seq, dtype=torch.float32, device=device)
            ftrs.append(traj)
            lengths.append(traj.size(0))
        
        padded = nn.utils.rnn.pad_sequence(ftrs, batch_first=True)
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=device)
        return padded, lengths

    def featurize_vehicle(self, vehs):
        """Build terminal vehicle features."""
        # prepare
        horizon = max(RH.T - RH.t0, 1e-6)
        
        ftrs = []
        for k in vehs:
            route = RH.sol[k]
            tail_node = int(route[-1,0])
            tail_coord = RH.get_node_coord(tail_node)
            xy = RH.normalize_coord(tail_coord)
            tail_time = float(route[-1,2])
            ftrs.append([xy[0], xy[1], (tail_time - RH.t0) / horizon])
        
        return np.asarray(ftrs, dtype=np.float32)

    def save_model(self, save_path, backup=False):
        '''save crucial model parameters'''
        save = {
            # neural networks
            'request_view': self.request_view.state_dict(),
            'vehicle_view': self.vehicle_view.state_dict(),
            'context_view': self.context_view.state_dict(),
            'actor': self.actor.state_dict(),
            'dummy_vehicle_x': self.dummy_vehicle_x.detach().cpu(),
            
            # optimizers
            'actor_opti': self.actor_opti.state_dict(),

            # records
            'steps': steps, 
            'episode': self.episode, 
            'returns': self.returns, 
            'runtime': self.runtime,
            'ppo_clip': self.ppo_clip,
            'ppo_epochs': self.ppo_epochs,
            'ppo_batch_size': self.ppo_batch_size,
            'entropy_coef': self.entropy_coef}
        
        torch.save(save, save_path)
        torch.save(save, save_path.replace('.pth', '_bkp.pth'))
    
    def load_model(self, load_path=None):
        if load_path is None:
            return
        
        '''load model parameters'''
        params = torch.load(load_path, map_location=device, weights_only=False)
        # load model parameters
        self.request_view.load_state_dict(params['request_view'])
        self.vehicle_view.load_state_dict(params['vehicle_view'])
        self.context_view.load_state_dict(params['context_view'])
        self.actor.load_state_dict(params['actor'])
        self.dummy_vehicle_x.data.copy_(params['dummy_vehicle_x'].to(device))
        
        self.actor_opti.load_state_dict(params['actor_opti'])
        
        
    def ARS_temperature_control(self):
        '''ARS adpative temperature'''
        # (1) adjust temperature (heat/cool)
        if self.heat_gate:
            self.τ = min(self.τ + self.heat_coef, self.τ_max)
        else:
            self.τ = max(self.τ - self.heat_coef, self.τ_min)
        
        # (2) switch gate if needed
        if (self.τ >= self.τ_max) or (self.τ <= self.τ_min):
            self.temp_fail += 1
    
        if (self.τ >= self.τ_max) and (self.temp_fail >= self.switch_cap):
            print('cool!')
            self.heat_gate = False
            self.temp_fail = 0
            self.heat_coef = (self.τ_max - self.τ_min) / self.episode
        elif (self.τ <= self.τ_min) and (self.temp_fail >= self.switch_cap):
            print('heat!')
            self.heat_gate = True
            self.temp_fail = 0
            self.heat_coef = (self.τ_max - self.τ_min) / self.episode
            
    def train(self, data, early_breaking=True, tolerance=250, per_epis_depict=1000):
        recent_fail = 0
        start = time.time()
        for episode in range(self.max_episode): # do an episode
            # estabilsh rolling horizon model
            global RH
            RH = RollingHorizon(self, data, vehi_depot, steps=steps)
            ov_epi = RH.main()
            self.returns[episode] = ov_epi
            
            # record / ARS temperature control
            if ov_epi > self.ov_best:
                self.ov_best = ov_epi
            else:
                self.fail += 1
                #RL.temp_fail += 1
                if self.ARS:
                    self.ARS_temperature_control()
                
            # details
            if episode > 0:
                recent_returns = self.returns[max(self.episode-self.recent, 0
                                                  ):self.episode]
                recent_mean = recent_returns.mean()
            else:
                recent_mean = -np.inf
            
            print(f'Instance [{inst.replace(".parquet", "")}] {inst_idx + 1}/{len(insts)}')
            print(f'Episode {self.episode}/{self.max_episode}  Temperature {self.τ:.2f}')
            print(f'Incumbent/Best ov [{ov_epi:.3f} / {self.ov_best:.3f}]')
            print(f'recent {self.recent} average: {recent_mean:.3f}')

            # turn off ARS when stablized
            if self.ARS and (self.episode > 1000) and \
                            (recent_mean >= self.off_cap):
                self.τ = 0
                self.ARS = False
            
            # depict reward 
            if self.episode % per_epis_depict == 0:
                plt.plot(np.arange(self.max_episode), 
                         self.returns, color='tab:red')
                plt.show()
                plt.close()
            
            # check breaking
            if recent_mean > self.recent_best:
                self.recent_best = recent_mean
                end = time.time(); self.runtime = end - start
                self.save_model(save_path=save_path, 
                                backup=False)  # save the best recent model
                recent_fail = 0
            else:
                recent_fail += 1
                if early_breaking and episode > 1000 and recent_fail >= tolerance:   # early breaking
                    break

            self.episode += 1
            print()
        end = time.time()
        self.runtime += end - start
        
    def evaluate(self, data, trys=1, greedy_sample=False):
        '''evaluate model with several trys'''
        returns = np.full(trys, np.nan)
        self.request_view.eval(); self.vehicle_view.eval(); self.context_view.eval()
        self.actor.eval()
        
        for i in range(trys):    # an evaluation
            global RH
            RH = RollingHorizon(self, data, vehi_depot, steps=steps) 
            ov_i = RH.main(train=False, greedy_sample=greedy_sample)
            returns[i] = ov_i
        return returns

        self.request_view.train(); self.vehicle_view.train(); self.context_view.train()
        self.actor.train()

# ===============================
# Dynamic Rolling Horizon Model
# ===============================
class RollingHorizon():
    def __init__(self, env, data, vehi_depot, steps=5):
        self.RL = env
        # data
        self.data = data.copy()
        
        # time settings
        self.t0 = float(self.data['pick_early'].min())
        self.T = float(self.data['pick_late'].max())
        self.Δt = float(self.T - self.t0) / max(steps-1, 1)
        self.now = self.t0
        
        # node maps
        self.depot_map = {-(d+1): depots[d] for d in range(len(depots))}
        self.req_map = self.data.set_index('id', drop=False)
        self.fare_map = dict(zip(self.data['id'],  self.data['fare']))

        # solution
        self.sol = [np.array([[vehi_depot[k], -1, self.t0, 0]], 
                             dtype=float) for k in range(K)]
        self.heads = np.zeros(K, dtype=int)  # heads of vehicle availale node
        self.ov = 0.0
    
    def main(self, train=True, greedy_sample=False): 
        '''rolling horizon'''
        while self.now <= self.T:
            self.time_pass()   # solution end's time is adjusted to now 
            self.update_heads()
            It = self.get_active_requests()
            
            for _, req in It.iterrows():
                avail_vehs = np.arange(K)
                if len(avail_vehs) == 0:
                    break
                
                car_idx, eta, transition = self.RL.act(
                    req, range(K), greedy_sample=greedy_sample)
                is_dummy = True if car_idx == K else False
                applied, Δov = self.apply(req, car_idx, eta, is_dummy=is_dummy)
                self.update_heads()

                # back propagation
                if train:
                    self.RL.update(transition, Δov, is_dummy=is_dummy)
            self.now += self.Δt  
        if train:
            self.RL.ppo_update()
        return self.ov
    
    def get_time_tonext_available(self):
        '''get remaining time to vehicles' next available time'''
        time_arr = np.array([self.sol[k][-1,2] for k in range(K)])
        return time_arr - self.now 
        
    def get_avilable_vehicle(self):
        '''get available vehicles without dispatch task'''
        time_arr = np.array([self.sol[k][-1,2] for k in range(K)])
        avail_vehs = np.arange(K)[time_arr <= self.now]
        return time_arr, avail_vehs
    
    def time_pass(self):
        '''solution end's time is adjusted to now if it's earlier'''
        for k, route in enumerate(self.sol):
            end_time = route[-1,2]
            self.sol[k][-1,2] = max(end_time, self.now)

    def apply(self, req, car_idx, eta, is_dummy=True):
        """Append the request to the selected vehicle route. """
        applied = False
        Δov = 0.0

        # dummy vehicle: reject / no dispatch for now
        if is_dummy:
            return applied, Δov

        route = self.sol[car_idx]
        
        # request 
        req_id = int(req['id']); eta = int(eta)
        pick_coord = np.array([req['pick_longt'], req['pick_lat']])
        fare = req['fare']
        
        # distance
        # vehicle to pickup (note: since vehicle-to-pick has not actual record, 
                            #      we use average speed approximately)
        prev_id = int(route[-1,0])
        prev_coord = self.get_node_coord(prev_id)
        dis_topick = self.get_distance(prev_coord, pick_coord)
        t_topick = dis_topick / avg_v
        
        # pick to drop
        dis_prev = route[-1,3]
        dis_trip = req['trip_dis']
        t_trip = dis_trip / req['speed']  # actual spatiotemporal speed recorded 
        Δdis = dis_topick + dis_trip
        
        # calculate profit
        t_prev = route[-1,2]
        t_pick = max(float(req['pick_early']), t_prev + t_topick)
        t_drop = t_pick + t_trip
        Δtime = t_drop - t_prev
        
        # ETA impact factor
        if t_drop < float(req['drop_early']):
            advance = (float(req['drop_early']) - t_drop) * 60
            coefs = np.clip(advance * reward_coefs[0], None, 1)
        elif t_drop > float(req['drop_late']):
            lateness = (t_drop - float(req['drop_late'])) * 60
            coefs = np.clip(lateness * reward_coefs[2], -1, None)
        else:
            coefs = reward_coefs[1]
        
        # profit (service fare - mileage cost - time cost)
        Δov = float(fare * (1 + coefs[eta]) - (μm*Δdis + μt*Δtime))  
        
        # apply
        new_node = [[req_id, eta, t_drop, dis_prev + Δdis]]
        self.sol[car_idx] = np.vstack([route, new_node])
        self.ov += Δov
        self.data.loc[self.data['id'] == req_id, 'status'] = False
        self.req_map.loc[req_id, 'status'] = False
        return True, Δov

    def calculate_ov(self):
       '''calculate objective value, this function can be used to validate'''
       ov = 0.0
       for k in range(K):
           route = self.sol[k]
           if len(route) <= 1:
               return ov
           
           for idx in range(1,len(route)):
               node = route[idx]
               prev_node = route[idx - 1]
               
               node_id = int(node[0])
               eta = int(node[1])
           
               req = self.req_map.loc[node_id]
               pick_coord = np.array([req['pick_longt'], req['pick_lat']])
               fare = req['fare']
           
               # previous node -> current request pickup
               prev_id = int(prev_node[0])
               prev_coord = self.get_node_coord(prev_id)
               dis_topick = self.get_distance(prev_coord, pick_coord)
               t_topick = dis_topick / avg_v
           
               # pickup -> drop-off
               dis_trip = req['trip_dis']
               speed = req['speed'] if req['speed'] > 0 else avg_v
               t_trip = dis_trip / speed
           
               Δdis = dis_topick + dis_trip
           
               # time
               t_prev = prev_node[2]
               if not np.isfinite(t_prev):
                   t_prev = self.t0
           
               t_pick = max(req['pick_early'], t_prev + t_topick)
               t_drop = t_pick + t_trip
               Δtime = t_drop - t_prev
           
               # ETA impact factor
               if t_drop < req['drop_early']:
                   advance = (req['drop_early'] - t_drop) * 60
                   coefs = np.clip(advance * reward_coefs[0], None, 1)
               elif t_drop > req['drop_late']:
                   lateness = (t_drop - req['drop_late']) * 60
                   coefs = np.clip(lateness * reward_coefs[2], -1, None)
               else:
                   coefs = reward_coefs[1]
           
               Δov = float(fare * (1 + coefs[eta]) - (μm * Δdis + μt * Δtime))
               ov += Δov
       return ov

    def update_heads(self):
        '''update the index of earliest vehicle available node''' 
        for idx, route in enumerate(self.sol):
            route = np.asarray(route, dtype=float)
            time_arr = route[:,2]
            head = np.where(np.isfinite(time_arr) & (time_arr <= self.now))[0]
            self.heads[idx] = int(head[-1]) if len(head) > 0 else 0
            
    def get_active_requests(self, sort_fare=True):
        '''get active requests at current time'''
        It = self.data[(self.data['pick_early'] <= self.now) 
                       & (self.data['pick_late'] >= self.now)
                       & (self.data['status'])].copy()
        
        if sort_fare and len(It) > 0:
            It = It.sort_values(by='fare', ascending=False)
        return It.reset_index(drop=True)
    
    def get_node_coord(self, node_id):
        '''get node's (longitude, latitude)'''
        node_id = int(node_id)
        if node_id < 0:
            return self.depot_map[node_id]
        req = self.req_map.loc[node_id]
        return [req['drop_longt'], req['drop_lat']]
    
    @staticmethod
    def get_distance(coord1, coord2):
        '''get haversine distance between two coordinates on earth'''
        longt1, lat1 = np.radians(coord1[0]), np.radians(coord1[1])
        longt2, lat2 = np.radians(coord2[0]), np.radians(coord2[1])

        Δlongt = longt1 - longt2
        Δlat = lat1 - lat2

        a = (np.sin(Δlat / 2) ** 2
             + np.cos(lat1) * np.cos(lat2) * np.sin(Δlongt / 2) ** 2)
        return float(2 * np.arcsin(np.sqrt(a)) * 6371)
    
    @staticmethod
    def normalize_coord(coord):
        '''Convert longitude/latitude into local relative coordinates. '''
        x = (coord[0] - longt0) * np.cos(np.radians(lat0))
        y = coord[1] - lat0
        return np.array([x, y], dtype=np.float32)
    

# ===============================
# Parameters
# ===============================

# initial depots 
# assume index of depot[0] is -1, index of depot[1] is -2
depots = np.array([
    [-73.9855, 40.7580],  # Times Square
    [-74.0090, 40.7060],  # Wall Street / Financial District
    [-73.9780, 40.7527],  # Grand Central Terminal
    [-73.9934, 40.7505],  # Penn Station
    [-73.9776, 40.7614],  # Rockefeller Center
    [-73.9680, 40.7851],  # Central Park East
    [-73.9819, 40.7681],  # Columbus Circle
    [-73.9830, 40.7420],  # Chelsea
    [-73.9500, 40.7230],  # Williamsburg (Brooklyn)
    [-73.9442, 40.6782],  # Downtown Brooklyn
    [-73.8300, 40.7580],  # Flushing (Queens)
    [-73.8656, 40.7681],  # Jackson Heights
    [-73.7781, 40.6413],  # JFK Airport
    [-73.8740, 40.7769],  # LaGuardia Airport
    [-73.9505, 40.8055],  # Harlem
    [-73.9955, 40.7309],  # Greenwich Village
    [-74.0150, 40.7113],  # World Trade Center
    [-73.9397, 40.7003],  # Bushwick (Brooklyn)
    [-73.9763, 40.6442],  # Prospect Park South
    [-73.9875, 40.7484]   # Empire State Building
])

K = 20                          # fleet size
avg_v = 12.94                   # default average speed (according to statistic)
longt0 = -73.96; lat0 = 40.74   # normalized coordinate center
μm = 1.25; μt = 5.0             # distance-to-cost, time-to-cost coefficients
steps = 20                       # optimization steps
vehi_depot = {k:-((k%len(depots)) + 1) for k in range(K)}  # vehicle depot


# ETA-impact factor matrix
# ETA → early, on-time, late    
reward_coefs = np.array([ # ATA ↓
    [0.2, 0.10, 0.05],    # early arrival
    [-0.15, 0.0, -0.15],    # on-time arrival
    [-0.2, -0.10, -0.05]  # late arrival
])


# ===============================
# Paths
# ===============================

ROOT_DIR = Path(__file__).resolve().parents[2]
inst_dir = ROOT_DIR / 'data' / 'Training' / 'VS-size'
save_dir = ROOT_DIR / 'asset'
insts = sorted(inst_dir.glob('*.parquet'))
print(insts)


if __name__ == '__main__':
    save_dir.mkdir(parents=True, exist_ok=True)

    for inst_idx, inst_path in enumerate(insts):  # patch training instances in dataset
        # RL save & load paths
        save_path = str(save_dir / inst_path.name.replace('.parquet', '_PPO.pth'))

        ins = Instance(inst_path)
        RL = Learning(max_episode=5000, γ=0.5, reward_scale=0.01, 
                      τ=1.0, τ_max=10.0, τ_min=0.0, reject_loss=-10, 
                      lr_actor=1e-4, ppo_clip=0.2, ppo_epochs=4,
                      ppo_batch_size=64, entropy_coef=0.0)
        RL.train(data=ins.data, early_breaking=False)
        RL.save_model(save_path=save_path, backup=True)
