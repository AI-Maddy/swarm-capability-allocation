#!/usr/bin/env python3
"""
mission.py -- kinematic mission simulator coupled to the latency model.

M1 (stressed locomotion): agents wander by random waypoint at high density;
    the reflex command applied at step t is the one computed from the snapshot
    at t - l, with l drawn per agent per step from the latency distribution of
    the hosting architecture at the same swarm size (swarmlat.py).  A command
    that never arrives (deadline-exhausting drop) leaves the agent on its
    navigation velocity: the off-agent architectures carry no local reflex.
M2 (allocation mission): fixed arena and targets, matched sensing budget,
    specialist proportion swept; local reflex everywhere.

Reflex law (identical for every architecture; only where/when it is computed
differs): guard g_i = [some neighbour within r_p is closing, i.e. negative
range rate]; response = withdraw at full speed along the unit vector of
sum_j (x_i - x_j)/|x_i - x_j|^2 over closing neighbours within r_p.  A collision is the onset of a contact |x_i - x_j| < 2 r_c.
"""
import math, os, sys, csv, time
import numpy as np
from scipy.spatial import cKDTree
import swarmlat as SL

M = dict(v=8.0, r_p=2.5, r_c=0.4, r_e=8.0, dt=0.002, T=12.0, warm=2.0,
         rho=0.40, lag_cap=0.5)


# ------------------------------------------------------------ latency samples
def latency_samples(arch, N, rho, seed):
    p = dict(SL.P); p['rho'] = rho; p['events'] = 20000
    rng = np.random.default_rng(seed)
    if arch == 'L':
        d = SL.sim_local(N, 0.0, p, rng)
    elif arch == 'E':
        d = SL.sim_onehop(N, 0.0, p, rng)
    elif arch == 'S-ALOHA':
        d = SL.sim_sink(N, 0.0, p, rng, 'aloha', 1)
    elif arch.startswith('S-TDMA-'):
        d = SL.sim_sink(N, 0.0, p, rng, 'tdma', int(arch.split('-')[-1]))
    else:
        raise ValueError(arch)
    return np.asarray(d)


# ------------------------------------------------------------ geometry helpers
def jittered_grid(N, side, rng, jitter=0.2):
    n = math.ceil(math.sqrt(N))
    g = (np.arange(n) + 0.5) * side / n
    xx, yy = np.meshgrid(g, g)
    pts = np.c_[xx.ravel(), yy.ravel()][:N]
    return pts + rng.uniform(-jitter, jitter, pts.shape) * side / n


def repulsion(pos, r, box=None, vel=None):
    """Reflex guard/response from one snapshot.  A neighbour within r triggers
    the guard only if the pair is closing (negative range rate), which a
    proximity sensor measures locally.  Returns repulsion, guard, all pairs
    within r and their distances (for contact detection)."""
    tree = cKDTree(pos, boxsize=box)
    pr = tree.query_pairs(r, output_type='ndarray')
    rep = np.zeros_like(pos)
    guard = np.zeros(len(pos), bool)
    if pr.size == 0:
        return rep, guard, pr, np.empty(0)
    dv = pos[pr[:, 0]] - pos[pr[:, 1]]
    if box is not None:
        dv -= box * np.round(dv / box)
    d2 = np.maximum((dv ** 2).sum(1), 1e-6)
    act = np.ones(len(pr), bool)
    if vel is not None:
        dvel = vel[pr[:, 0]] - vel[pr[:, 1]]
        act = (dv * dvel).sum(1) < 0.0
    w = dv[act] / d2[act, None]
    a, b = pr[act, 0], pr[act, 1]
    np.add.at(rep, a, w)
    np.add.at(rep, b, -w)
    guard[a] = True; guard[b] = True
    return rep, guard, pr, np.sqrt(d2)


def unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), 0.0)


# ------------------------------------------------------------ M1 locomotion
def run_m1(arch, N, seed, m=M):
    rng = np.random.default_rng(seed)
    lat = latency_samples(arch, N, m['rho'], seed + 7)
    side = math.sqrt(N / m['rho'])
    dt = m['dt']; v = m['v']
    pos = np.mod(jittered_grid(N, side, rng), side)
    wp = rng.uniform(0, side, (N, 2))
    L = int(m['lag_cap'] / dt) + 1
    rep_h = np.zeros((L, N, 2), np.float32)
    grd_h = np.zeros((L, N), bool)
    steps = int(m['T'] / dt); warm = int(m['warm'] / dt)
    prev = set(); onsets = 0; reflex_applied = 0; guard_true = 0
    lag_steps_all = np.where(np.isfinite(lat), np.ceil(lat / dt), np.inf)
    vel = np.zeros_like(pos)
    for s in range(steps):
        rep, grd, pr, dist = repulsion(pos, m['r_p'], side, vel)
        rep_h[s % L] = rep; grd_h[s % L] = grd
        # contacts
        if pr.size:
            c = pr[dist < 2 * m['r_c']]
            cur = set((c[:, 0].astype(np.int64) * N + c[:, 1]).tolist())
        else:
            cur = set()
        if s >= warm:
            onsets += len(cur - prev)
            guard_true += int(grd.sum())
        prev = cur
        # stale command
        lag = lag_steps_all[rng.integers(0, lag_steps_all.size, N)]
        ok = np.isfinite(lag) & (lag < L) & (lag <= s)
        idx = np.zeros(N, int)
        idx[ok] = ((s - lag[ok]) % L).astype(int)
        ag = np.nonzero(ok)[0]
        use = np.zeros(N, bool)
        use[ag] = grd_h[idx[ag], ag]
        # navigation (minimum-image direction on the torus)
        to = wp - pos; to -= side * np.round(to / side)
        dwp = np.linalg.norm(to, axis=1)
        new = dwp < 1.0
        if new.any():
            wp[new] = rng.uniform(0, side, (new.sum(), 2))
            to = wp - pos; to -= side * np.round(to / side)
        vel = v * unit(to)
        if use.any():
            vel[use] = v * unit(rep_h[idx[use], np.nonzero(use)[0]].astype(float))
            if s >= warm:
                reflex_applied += int(use.sum())
        pos = np.mod(pos + vel * dt, side)
    minutes = (steps - warm) * dt / 60
    return dict(exp='M1', arch=arch, N=N, seed=seed,
                coll_rate=2 * onsets / (N * minutes),
                onsets=onsets,
                reflex_frac=reflex_applied / max(guard_true, 1),
                miss_inf=float(np.mean(~np.isfinite(lat))))


# ------------------------------------------------------------ M2 allocation
def run_m2(mode, f, seed, side=70.0, n_targets=30, budget=900, T=120.0,
           dt=0.01, service=5.0, detect_delay=0.5, m=M):
    """mode 'C' (substrate+organs), 'B' (partitioned), 'A' (generalist)."""
    rng = np.random.default_rng(seed)
    if mode == 'A':
        N = budget // 101; n_ext = N
    elif mode == 'C':
        N = int(budget // (1 + 100 * f)); n_ext = max(1, round(f * N))
    else:  # B: specialists cost 100, have no proximity and cannot move
        n_ext = max(1, round(f * budget / (1 + 100 * f)))
        N = n_ext + int(budget - 100 * n_ext)
    ext = np.zeros(N, bool); ext[:n_ext] = True
    mobile = np.ones(N, bool)
    prox = np.ones(N, bool)
    if mode == 'B':
        mobile[ext] = False; prox[ext] = False
    pos = jittered_grid(N, side, rng)
    wp = rng.uniform(0, side, (N, 2))
    tgt = rng.uniform(0, side, (n_targets, 2))
    t_det = np.full(n_targets, np.inf)       # time detected
    assigned = np.full(n_targets, -1)
    done = np.zeros(n_targets, bool)
    task = np.full(N, -1)                    # target index per agent
    serv_left = np.zeros(N)
    msgs = 0; onsets = 0; prev = set()
    vel_prev = np.zeros((N, 2))
    steps = int(T / dt)
    for s in range(steps):
        t = s * dt
        # detection by exteroceptive agents
        und = np.nonzero(~np.isfinite(t_det))[0]
        if und.size:
            dd = np.linalg.norm(tgt[und][:, None, :] - pos[ext][None, :, :], axis=2)
            hit = und[(dd < m['r_e']).any(1)]
            if hit.size:
                t_det[hit] = t; msgs += hit.size
        # assignment through shared memory after L3 delay
        ready = np.nonzero((t_det + detect_delay <= t) & (assigned < 0) & ~done)[0]
        for q in ready:
            idle = np.nonzero((task < 0) & mobile)[0]
            if idle.size == 0:
                break
            a = idle[np.argmin(np.linalg.norm(pos[idle] - tgt[q], axis=1))]
            task[a] = q; assigned[q] = a; msgs += 1
        # navigation
        goal = wp.copy()
        busy = task >= 0
        goal[busy] = tgt[task[busy]]
        to = goal - pos
        dist = np.linalg.norm(to, axis=1)
        reach_wp = (~busy) & (dist < 1.0)
        if reach_wp.any():
            wp[reach_wp] = rng.uniform(0, side, (reach_wp.sum(), 2))
        arrived = busy & (dist < 0.5)
        serv_left[arrived] += dt
        fin = arrived & (serv_left >= service)
        for a in np.nonzero(fin)[0]:
            done[task[a]] = True; task[a] = -1; serv_left[a] = 0; msgs += 1
        vel = m['v'] * unit(to)
        vel[arrived] = 0.0
        # local reflex (every architecture here hosts it locally)
        rep, grd, pr, dd2 = repulsion(pos, m['r_p'], None, vel_prev)
        # agents without proximity sensing cannot run the reflex
        grd &= prox
        use = grd & mobile
        vel[use] = m['v'] * unit(rep[use])
        vel[~mobile] = 0.0
        if pr.size:
            c = pr[dd2 < 2 * m['r_c']]
            cur = set((c[:, 0].astype(np.int64) * N + c[:, 1]).tolist())
        else:
            cur = set()
        if s > 0:
            onsets += len(cur - prev)
        prev = cur
        pos = np.clip(pos + vel * dt, 0, side)
        vel_prev = vel
        if done.all():
            break
    T_used = (s + 1) * dt
    return dict(exp='M2', mode=mode, f=f, seed=seed, N=N, n_ext=n_ext,
                completion=float(done.mean()), t_end=T_used,
                coll_rate=2 * onsets / (N * T_used / 60),
                msg_rate=msgs / (N * T_used),
                energy_baseline_J=0.08 * N * T)


# ------------------------------------------------------------ driver
def _m1(args):
    return run_m1(*args)


def _m2(args):
    return run_m2(*args)


def main():
    from multiprocessing import Pool
    os.makedirs('results', exist_ok=True)
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('all', 'm1'):
        jobs = [(a, N, s) for N in [3000, 1000, 600, 300, 100]
                for a in ['L', 'E', 'S-ALOHA', 'S-TDMA-1', 'S-TDMA-4']
                for s in [1, 2, 3]]
        t0 = time.time()
        with Pool(2) as pool:
            res = []
            for r in pool.imap_unordered(_m1, jobs):
                print(r, f'{time.time()-t0:.0f}s', flush=True); res.append(r)
        keys = ['exp', 'arch', 'N', 'seed', 'coll_rate', 'onsets', 'reflex_frac', 'miss_inf']
        with open('results/mission_m1.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in sorted(res, key=lambda r: (r['arch'], r['N'], r['seed'])):
                w.writerow(r)
    if which in ('all', 'm2'):
        fs = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
        jobs = [('C', f, s) for f in fs for s in range(1, 6)] + \
               [('B', f, s) for f in fs for s in range(1, 6)] + \
               [('A', 1.0, s) for s in range(1, 6)]
        t0 = time.time()
        with Pool(2) as pool:
            res = []
            for r in pool.imap_unordered(_m2, jobs):
                print(r, f'{time.time()-t0:.0f}s', flush=True); res.append(r)
        keys = ['exp', 'mode', 'f', 'seed', 'N', 'n_ext', 'completion', 't_end',
                'coll_rate', 'msg_rate', 'energy_baseline_J']
        with open('results/mission_m2.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
            for r in sorted(res, key=lambda r: (r['mode'], r['f'], r['seed'])):
                w.writerow(r)


if __name__ == '__main__':
    main()
