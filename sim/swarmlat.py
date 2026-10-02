#!/usr/bin/env python3
"""
swarmlat.py -- reflex-loop latency simulator for capability-allocation study.

Hosting architectures for a reflex loop (sense -> compute -> act):
  L        : local hosting (criterion-compliant). No network.
  E        : one-hop hosting. Sensor data sent to a helper within one hop,
             computed there, command returned. Two transmissions per event,
             contention in a local interference disk (mean-field slotted ALOHA).
  S-ALOHA  : sink-directed hosting, random access at the sink (slotted ALOHA
             with timeout + randomised backoff), slot-level discrete-event sim.
  S-TDMA-k : sink-directed hosting, ideal collision-free schedule at the sink
             with k parallel receivers (best case for any MAC), slot-level DES.

Outer hops (all hops except the one into/out of the sink) are modelled as
contention-free (airtime + channel-loss retries only).  Relay queueing is
ignored.  Both simplifications favour the sink-directed architectures.

Usage:  python3 swarmlat.py            -> writes results/*.csv
"""
import math, os, csv
import numpy as np

# ---------------------------------------------------------------- parameters
P = dict(
    Ts=2.0e-3,      # sensing period (s); sensing age ~ U(0, Ts)
    Tc=0.3e-3,      # reflex compute time (s)
    Ta=1.2e-3,      # airtime of one reflex packet = one MAC slot (s)
    To=8.0e-3,      # retransmission timeout (s)
    max_retx=4,     # retries after first attempt; exhausting them = miss
    lam=2.0,        # reflex event rate per agent (Hz)
    rho=0.25,       # areal density (agents / m^2)
    R=15.0,         # radio range (m) -> hop length
    r_int=5.0,      # interference radius of a short one-hop link (m)
    backoff_slots=4,# randomised backoff after timeout, U{0..backoff_slots-1}
    events=40000,   # target events per condition
)


def sensing_age(rng, n, p):
    return rng.uniform(0.0, p['Ts'], n)


# ------------------------------------------------------------------- L, E
def sim_local(N, ploss, p, rng):
    n = p['events']
    return sensing_age(rng, n, p) + p['Tc']


def one_hop_success(ploss, p):
    """Mean-field fixed point for per-attempt success in a local disk."""
    n_c = p['rho'] * math.pi * p['r_int'] ** 2
    G0 = n_c * p['lam'] * 2 * p['Ta']           # offered load, first attempts
    s = 1.0
    for _ in range(200):
        q = 1 - s
        m = p['max_retx']
        Eatt = (1 - q ** (m + 1)) / (1 - q) if q < 1 else m + 1
        G = G0 * Eatt
        s_new = (1 - ploss) * math.exp(-G)
        if abs(s_new - s) < 1e-12:
            break
        s = 0.5 * s + 0.5 * s_new
    return s, G


def sim_onehop(N, ploss, p, rng):
    n = p['events']
    s, _ = one_hop_success(ploss, p)
    d = sensing_age(rng, n, p) + p['Tc']
    for _leg in range(2):                       # uplink + downlink
        fails = rng.geometric(s, n) - 1         # failed attempts before success
        d = d + p['Ta'] + fails * p['To']
        d[fails > p['max_retx']] = np.inf
    return d


# ---------------------------------------------------------- sink-directed DES
def _outer_hops_delay(h_extra, ploss, p, rng):
    """Contention-free hops: airtime + timeout per channel loss."""
    tot = np.zeros(h_extra.shape)
    hmax = int(h_extra.max()) if h_extra.size else 0
    for k in range(hmax):
        act = h_extra > k
        if ploss > 0:
            fails = rng.geometric(1 - ploss, act.sum()) - 1
            add = p['Ta'] + fails * p['To']
            bad = fails > p['max_retx']
            add[bad] = np.inf
        else:
            add = p['Ta']
        tot[act] += add
    return tot


def sim_sink(N, ploss, p, rng, mac='aloha', k=1):
    """Slot-level DES of the sink domain.  Returns per-event reflex delay."""
    lam_tot = N * p['lam']
    n_ev = p['events']
    Ta = p['Ta']
    T = n_ev / lam_tot                       # simulated horizon (s)
    warm = 0.2 * T
    n_tot = rng.poisson(lam_tot * (T + warm))
    t0 = np.sort(rng.uniform(0, T + warm, n_tot))
    # positions -> hop count
    rN = math.sqrt(N / (math.pi * p['rho']))
    dist = rN * np.sqrt(rng.uniform(0, 1, n_tot))
    h = np.maximum(1, np.ceil(dist / p['R'])).astype(int)
    age = sensing_age(rng, n_tot, p)
    up_outer = _outer_hops_delay(h - 1, ploss, p, rng)
    dn_outer = _outer_hops_delay(h - 1, ploss, p, rng)

    ready = t0 + up_outer                    # time packet reaches sink domain
    done = np.full(n_tot, np.inf)

    # packet table (uplink + downlink share it); grow as needed
    cap = 4 * n_tot + 16
    pk_ev = np.empty(cap, int); pk_next = np.empty(cap); pk_att = np.empty(cap, int)
    pk_dir = np.empty(cap, int)               # 0 uplink, 1 downlink
    alive = np.zeros(cap, bool)
    n_pk = 0
    order = np.argsort(ready)
    order = order[np.isfinite(ready[order])]      # outer-hop drops never arrive
    ready_sorted = ready[order]
    n_tot_eff = order.size
    ptr = 0
    slot = 0
    t_end = ready.max() if np.isfinite(ready).any() else T + warm
    max_slots = int((T + warm) / Ta * 3) + 2000
    while slot < max_slots:
        t = slot * Ta
        # admit uplink packets that reached the sink domain
        j = np.searchsorted(ready_sorted, t, side='right')
        if j > ptr:
            idx = order[ptr:j]
            idx = idx[np.isfinite(ready[idx])]
            m = idx.size
            pk_ev[n_pk:n_pk + m] = idx; pk_next[n_pk:n_pk + m] = t
            pk_att[n_pk:n_pk + m] = 0; pk_dir[n_pk:n_pk + m] = 0
            alive[n_pk:n_pk + m] = True
            n_pk += m; ptr = j
        live = np.nonzero(alive[:n_pk])[0]
        if live.size == 0:
            if ptr >= n_tot_eff:
                break
            slot = max(slot + 1, int(ready_sorted[ptr] / Ta)); continue
        cand = live[pk_next[live] <= t + 1e-12]
        if cand.size:
            if mac == 'aloha':
                tx = cand
                ok_slot = (tx.size == 1)
                succ = tx if ok_slot else np.empty(0, int)
                fail = np.empty(0, int) if ok_slot else tx
                if succ.size and ploss > 0 and rng.random() < ploss:
                    fail = succ; succ = np.empty(0, int)
            else:  # ideal schedule, FIFO by readiness, k receivers per slot
                srt = cand[np.argsort(pk_next[cand], kind='stable')]
                tx = srt[:k]
                if ploss > 0:
                    lost = rng.random(tx.size) < ploss
                    succ, fail = tx[~lost], tx[lost]
                else:
                    succ, fail = tx, np.empty(0, int)
            # successes
            for q in succ:
                e = pk_ev[q]
                alive[q] = False
                if pk_dir[q] == 0:          # uplink done -> compute -> downlink
                    pk_ev[n_pk] = e; pk_next[n_pk] = t + Ta + p['Tc']
                    pk_att[n_pk] = 0; pk_dir[n_pk] = 1; alive[n_pk] = True
                    n_pk += 1
                else:
                    done[e] = t + Ta + dn_outer[e]
            # failures
            if fail.size:
                pk_att[fail] += 1
                dead = fail[pk_att[fail] > p['max_retx']]
                alive[dead] = False
                keep = fail[pk_att[fail] <= p['max_retx']]
                if mac == 'aloha':
                    pk_next[keep] = (t + p['To'] +
                                     rng.integers(0, p['backoff_slots'], keep.size) * Ta)
                else:
                    pk_next[keep] = t + Ta     # scheduled retry next slot
        if n_pk > cap - 4 * k - 8:            # compact table
            lv = np.nonzero(alive[:n_pk])[0]
            m = lv.size
            for arr in (pk_ev, pk_next, pk_att, pk_dir):
                arr[:m] = arr[lv]
            alive[:] = False; alive[:m] = True; n_pk = m
        slot += 1
        if ptr >= n_tot_eff and not alive[:n_pk].any():
            break
    d = age + (done - t0) + p['Tc'] * 0  # Tc already inside downlink start
    meas = (t0 >= warm) & (t0 < warm + T)
    return d[meas]


# ------------------------------------------------------------------ analysis
def summarize(d, taus):
    d = np.asarray(d)
    fin = d[np.isfinite(d)]
    out = dict(n=d.size,
               p99=float(np.quantile(d, 0.99, method='higher')) if d.size else np.nan,
               p99_succ=float(np.quantile(fin, 0.99, method='higher')) if fin.size else np.nan,
               drop=float(np.mean(~np.isfinite(d))))
    for tau in taus:
        out[f'miss_{tau*1e3:g}ms'] = float(np.mean(d > tau))
    return out


def analytic(p, k=1, f=1):
    loc = p['Ts'] + p['Tc']
    D1 = loc + 2 * p['Ta'] + f * p['To']
    Nstar = k / (2 * p['lam'] * p['Ta'])
    return dict(local=loc, D1_f0=loc + 2 * p['Ta'], D1=D1, Nstar=Nstar)


def N_floor(tau, p):
    """Largest N whose edge agent can meet tau over an ideal, empty network."""
    Hmax = math.floor((tau - p['Ts'] - p['Tc']) / (2 * p['Ta']))
    if Hmax < 1:
        return 0.0
    return math.pi * p['rho'] * (Hmax * p['R']) ** 2


def main():
    rng = np.random.default_rng(20261001)
    os.makedirs('results', exist_ok=True)
    taus = [5e-3, 10e-3, 20e-3, 50e-3, 100e-3, 1.0]
    Ns = [10, 30, 100, 300, 1000, 3000, 10000]
    archs = [('L', lambda N, pl, pp: sim_local(N, pl, pp, rng)),
             ('E', lambda N, pl, pp: sim_onehop(N, pl, pp, rng)),
             ('S-ALOHA', lambda N, pl, pp: sim_sink(N, pl, pp, rng, 'aloha', 1)),
             ('S-TDMA-1', lambda N, pl, pp: sim_sink(N, pl, pp, rng, 'tdma', 1)),
             ('S-TDMA-4', lambda N, pl, pp: sim_sink(N, pl, pp, rng, 'tdma', 4))]
    keys = ['exp', 'arch', 'N', 'ploss', 'k', 'n', 'p99', 'p99_succ', 'drop'] + \
           [f'miss_{t*1e3:g}ms' for t in taus]
    rows = []

    def rec(exp, name, N, ploss, k, d):
        s = summarize(d, taus)
        s.update(exp=exp, arch=name, N=N, ploss=ploss, k=k)
        rows.append(s)
        print(f"{exp:6s} {name:9s} N={N:6d} loss={ploss:.1f} p99={s['p99']*1e3:10.3f} ms"
              f" miss10={s['miss_10ms']:.3f} miss100={s['miss_100ms']:.3f} n={s['n']}",
              flush=True)

    # Exp 1: architectures x swarm size x channel loss
    for ploss in [0.0, 0.2]:
        for name, f in archs:
            for N in Ns:
                rec('main', name, N, ploss, 0, f(N, ploss, P))

    # Exp 2: saturation threshold of an ideal schedule, N/N* sweep (Theorem 2)
    pp = dict(P); pp['events'] = 20000
    for k in [1, 4]:
        Nstar = k / (2 * P['lam'] * P['Ta'])
        for x in [0.25, 0.5, 0.75, 0.9, 0.95, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]:
            N = int(round(x * Nstar))
            rec('sat', f'S-TDMA-{k}', N, 0.0, k, sim_sink(N, 0.0, pp, rng, 'tdma', k))

    # Exp 3: one-hop threshold (Theorem 1): miss vs deadline, lossless channel
    #   recorded in the main table via miss_* columns (E is N-independent)

    # Exp 4: stacked-in-favour sink (faster radio, 4 receivers, quarter traffic)
    pf = dict(P); pf['Ta'] = P['Ta'] / 3; pf['To'] = P['To'] / 4
    pf['lam'] = P['lam'] / 4; pf['max_retx'] = 8; pf['events'] = 20000
    for N in [1000, 3000, 10000, 20000, 30000, 50000]:
        rec('favour', 'S-TDMA-4', N, 0.0, 4, sim_sink(N, 0.0, pf, rng, 'tdma', 4))

    with open('results/latency.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for r in rows:
            w.writerow({kk: r[kk] for kk in keys})

    # miss-vs-deadline curves at N = 1000 for the figure
    tgrid = np.logspace(-3, 0, 61)
    curves = {}
    for name, f in archs:
        d = f(1000, 0.0, P)
        curves[name] = [float(np.mean(d > t)) for t in tgrid]
    with open('results/miss_vs_tau_N1000.csv', 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['tau_s'] + [a for a, _ in archs])
        for i, t in enumerate(tgrid):
            w.writerow([t] + [curves[a][i] for a, _ in archs])

    print('analytic:', analytic(P), 'one-hop success (loss 0):', one_hop_success(0.0, P))
    print('favour N*:', 4 / (2 * pf['lam'] * pf['Ta']))
    for tau in [10e-3, 100e-3]:
        print('N_floor', tau, N_floor(tau, P))


if __name__ == '__main__':
    main()
