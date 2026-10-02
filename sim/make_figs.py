#!/usr/bin/env python3
"""Figures from results/*.csv (run swarmlat.py first). Written to figures/."""
import csv, math, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from swarmlat import P, analytic

os.makedirs('figures', exist_ok=True)

plt.rcParams.update({'font.size': 7.5, 'font.family': 'serif', 'axes.linewidth': 0.6,
                     'lines.linewidth': 1.1, 'legend.fontsize': 6.5,
                     'legend.frameon': False, 'pdf.fonttype': 42})
C = {'L': '#1f5fbf', 'E': '#2a9d5c', 'S-ALOHA': '#d1495b',
     'S-TDMA-1': '#e07a1f', 'S-TDMA-4': '#7b4ab8'}
LBL = {'L': 'L (local, A3)', 'E': 'E (one-hop)', 'S-ALOHA': 'S-ALOHA (A4)',
       'S-TDMA-1': 'S-TDMA, $k{=}1$', 'S-TDMA-4': 'S-TDMA, $k{=}4$'}
MK = {'L': 'o', 'E': 'D', 'S-ALOHA': 's', 'S-TDMA-1': '^', 'S-TDMA-4': 'v'}

rows = list(csv.DictReader(open('results/latency.csv')))
for r in rows:
    for k in r:
        if k not in ('exp', 'arch'):
            r[k] = float(r[k])


def sel(**kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ---------------- Fig. 3: miss rate vs N (two deadlines), lossless channel
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.65), sharey=True)
for ax, tau in zip(axs, ['10', '100']):
    for a in ['L', 'E', 'S-ALOHA', 'S-TDMA-1', 'S-TDMA-4']:
        rr = sorted(sel(exp='main', arch=a, ploss=0.0), key=lambda r: r['N'])
        N = [r['N'] for r in rr]
        m = [max(r[f'miss_{tau}ms'], 1e-4) for r in rr]
        ax.plot(N, m, marker=MK[a], ms=2.6, color=C[a], label=LBL[a])
    for k, ls in [(1, ':'), (4, '--')]:
        ax.axvline(k / (2 * P['lam'] * P['Ta']), color='0.45', ls=ls, lw=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(5e-5, 2); ax.set_xlim(7, 1.4e4)
    ax.set_xlabel('swarm size $N$')
    ax.set_title(f'$\\tau={tau}$ ms', fontsize=7.5, pad=2)
    ax.set_yticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    ax.set_yticklabels(['$\\leq10^{-4}$', '$10^{-3}$', '$10^{-2}$', '$10^{-1}$', '1'])
axs[0].set_ylabel('deadline-miss rate')
axs[1].text(208 * 0.82, 2.2e-4, '$N^*_{1}$', ha='right', fontsize=6.5, color='0.3')
axs[1].text(833 * 1.12, 2.2e-4, '$N^*_{4}$', ha='left', fontsize=6.5, color='0.3')
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, bbox_to_anchor=(0.53, 1.13),
           handlelength=1.6, columnspacing=0.8)
fig.subplots_adjust(left=0.15, right=0.98, bottom=0.22, top=0.80, wspace=0.08)
fig.savefig('figures/fig_miss_vs_N.pdf', bbox_inches='tight')
plt.close(fig)

# ---------------- Fig. 4: (a) saturation collapse, (b) miss vs deadline
fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.7))
ax = axs[0]
series = [('S-TDMA-1', 'sat', 1, 1 / (2 * P['lam'] * P['Ta']), '$k{=}1$, $N^*{=}208$'),
          ('S-TDMA-4', 'sat', 4, 4 / (2 * P['lam'] * P['Ta']), '$k{=}4$, $N^*{=}833$'),
          ('S-TDMA-4', 'favour', 4, 1e4, 'stacked, $N^*{=}10^4$')]
cols = [C['S-TDMA-1'], C['S-TDMA-4'], '0.25']
for (a, e, k, Ns, lab), c in zip(series, cols):
    rr = sorted(sel(exp=e, arch=a, ploss=0.0), key=lambda r: r['N'])
    x = [r['N'] / Ns for r in rr]
    y = [r['miss_100ms'] for r in rr]
    ax.plot(x, y, marker='o', ms=2.4, color=c, label=lab)
ax.axvline(1.0, color='0.45', ls=':', lw=0.7)
ax.set_xscale('log'); ax.set_xlabel('$N/N^*$')
ax.set_ylabel('miss rate, $\\tau{=}100$ ms')
ax.set_ylim(-0.04, 1.06); ax.legend(loc='upper left', handlelength=1.2)
ax.set_title('(a) Theorem 2', fontsize=7.5, pad=2)

ax = axs[1]
mv = list(csv.DictReader(open('results/miss_vs_tau_N1000.csv')))
t = np.array([float(r['tau_s']) for r in mv]) * 1e3
for a, ls, lab in [('L', '-', 'L'), ('E', '-', 'E'), ('S-ALOHA', '-', 'S-ALOHA'),
                  ('S-TDMA-4', (0, (2, 1.5)), 'S-TDMA $k{=}4$')]:
    y = np.array([max(float(r[a]), 1e-4) for r in mv])
    ax.plot(t, y, color=C[a], ls=ls, label=lab, drawstyle='steps-post')
an = analytic(P)
for f in [0, 1, 2]:
    ts = (P['Ts'] + P['Tc'] + 2 * P['Ta'] + f * P['To']) * 1e3
    ax.axvline(ts, color=C['E'], ls=':', lw=0.7)
    ax.text(ts * 1.06, [1.6e-4, 4.5e-4, 1.6e-4][f], f'$\\tau^*_{f}$', fontsize=6, color=C['E'])
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(1, 1000); ax.set_ylim(5e-5, 2)
ax.set_xlabel('deadline $\\tau$ (ms)'); ax.set_ylabel('miss rate, $N{=}1000$')
ax.set_title('(b) Theorem 1', fontsize=7.5, pad=2)
ax.text(1.15, 3e-3, 'L', color=C['L'], fontsize=7)
ax.text(40, 6e-3, 'E', color=C['E'], fontsize=7)
ax.text(30, 0.25, 'S-ALOHA, S-TDMA ($k{=}4$)', color=C['S-TDMA-4'], fontsize=6)
fig.subplots_adjust(left=0.13, right=0.99, bottom=0.22, top=0.88, wspace=0.45)
fig.savefig('figures/fig_theorems.pdf', bbox_inches='tight')
plt.close(fig)

# ---------------- Fig. 5: mission-level collision rate vs N (M1)
import collections
m1 = list(csv.DictReader(open('results/mission_m1.csv')))
g = collections.defaultdict(list)
for r in m1:
    g[(r['arch'], int(r['N']))].append(float(r['coll_rate']))
fig, ax = plt.subplots(figsize=(3.5, 1.75))
floor = 0.02
Ns = [100, 300, 600, 1000, 3000]
for a in ['L', 'E', 'S-ALOHA', 'S-TDMA-1', 'S-TDMA-4']:
    mu = np.array([np.mean(g[(a, N)]) for N in Ns])
    sd = np.array([np.std(g[(a, N)]) for N in Ns])
    y = np.maximum(mu, floor)
    ax.plot(Ns, y, marker=MK[a], ms=2.8, color=C[a], label=LBL[a])
for k, ls in [(1, ':'), (4, '--')]:
    ax.axvline(k / (2 * P['lam'] * P['Ta']), color='0.45', ls=ls, lw=0.7)
ax.text(208 * 1.06, 0.035, '$N^*_1$', fontsize=6.5, color='0.3')
ax.text(833 * 1.06, 0.035, '$N^*_4$', fontsize=6.5, color='0.3')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_ylim(floor * 0.8, 2000)
ax.set_yticks([floor, 0.1, 1, 10, 100, 1000])
ax.set_yticklabels(['0', '$10^{-1}$', '1', '10', '$10^{2}$', '$10^{3}$'])
ax.set_xlabel('swarm size $N$'); ax.set_ylabel('collisions / agent / min')
ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), handlelength=1.5)
fig.subplots_adjust(left=0.14, right=0.66, bottom=0.22, top=0.97)
fig.savefig('figures/fig_collisions.pdf', bbox_inches='tight')
plt.close(fig)
print('figures written')
