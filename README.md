<h1 align="center">GameWAM:<br>A World Action Model for Video Games</h1>

<p align="center">
  <strong>Yuncheng Guo<sup>1</sup>, Zhanqiu Zhang<sup>2,†</sup>, Yiwen Guo<sup>3,†</sup>, Weijia Li<sup>4,†</sup></strong>
  <br>
  <sup>1</sup>Fudan University &nbsp;·&nbsp; <sup>2</sup>LIGHTSPEED &nbsp;·&nbsp; <sup>3</sup>Independent Researcher &nbsp;·&nbsp; <sup>4</sup>Tsinghua Shenzhen International Graduate School
  <br>
  <sup>†</sup>Corresponding authors
</p>

<p align="center">
  <a href="https://yunncheng.github.io/GameWAM/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project page"></a>
  <a href="https://arxiv.org/abs/2608.26200"><img src="https://img.shields.io/badge/Paper-arXiv%3A2608.26200-brightgreen" alt="arXiv paper"></a>
  <img src="https://img.shields.io/badge/Code-before%20Sep%2015%2C%202026-lightgrey" alt="Code release planned before Sep 15, 2026">
  <img src="https://img.shields.io/badge/Model-before%20Sep%2015%2C%202026-lightgrey" alt="Model release planned before Sep 15, 2026">
  <img src="https://img.shields.io/badge/Data-before%20Sep%2015%2C%202026-lightgrey" alt="Data release planned before Sep 15, 2026">
</p>

<p align="center">
  <a href="https://yunncheng.github.io/GameWAM/">
    <img src="assets/gamewam-overview.webp" alt="GameWAM overview" width="95%">
  </a>
</p>

## Overview

Modern video games combine rapidly changing first-person observations, persistent world state, and heterogeneous native controls. Existing game agents map visual and task context directly to actions but lack explicit world-dynamics modeling, while interactive game world models predict visual futures from supplied actions but do not directly serve as task policies.

GameWAM bridges these two directions. To our knowledge, it is the first **World Action Model for native closed-loop gameplay and GUI control**. GameWAM jointly generates future visual observations and executable keyboard–mouse trajectories through parallel visual and action generative processes with block-causal conditioning and flow matching. During online control, it predicts beyond the committed action horizon, executes only a short prefix, observes the resulting environment state, and replans.

## Release Plan

- [ ] Code: expected before Sep 15, 2026
- [ ] Model: expected before Sep 15, 2026
- [ ] Data: expected before Sep 15, 2026

## Highlights

- **A World Action Model for games.** GameWAM jointly generates future visual observations and executable native actions using parallel Video and Action DiTs, block-causal attention, and joint flow matching.
- **Unified heterogeneous native control.** Continuous camera or cursor motion and concurrent discrete controls share one Action-DiT flow. A learned per-action router selects gameplay- or GUI-specific predictions and continuous-action normalization at every action timestep.
- **Long-horizon block–cycle control.** Each planning unit predicts beyond the committed horizon, executes only a short action prefix, and replans from new observations. A bounded within-cycle KV cache and hierarchical cross-cycle visual history preserve temporal continuity under bounded memory.
- **Competitive closed-loop performance.** Across Minecraft and ViZDoom, GameWAM achieves competitive task performance; in Minecraft, it uses fewer executed native actions than the compared agents across all evaluated task categories.
- **Low-Frequency Action Source Imprinting.** We uncover LASI, a source-sensitivity failure mode in which low-frequency components of the sampled action source steer coarse generated camera motion and can accumulate into persistent directional bias when reused across replanning steps.

## Method

GameWAM models visual futures and native actions with parallel causal Video and Action DiTs. Visual, instruction, proprioceptive, and historical conditions are integrated through block-causal interaction, while the action process directly generates executable keyboard–mouse trajectories. For heterogeneous game interaction, gameplay and GUI control retain one physical action layout but use mode-specific prediction distributions and continuous-action statistics.

<p align="center">
  <img src="assets/gamewam-framework.webp" alt="GameWAM training and inference architecture" width="95%">
</p>

### Block–cycle autoregressive control

Dense visual–action sampling captures rapid ego-motion, transient targets, and brief interaction events, but a fixed token and cache budget then covers less interaction time. GameWAM decouples prediction from commitment: each planning unit predicts a longer action horizon, commits only a shorter execution block, and replans after receiving new observations.

Within a cycle, a bounded KV cache preserves fine-grained context. Across cycle boundaries, recent executed observations remain explicitly represented while older information is compressed into long-term hierarchical history.

<p align="center">
  <img src="assets/gamewam-schedule.webp" alt="GameWAM block-cycle control and hierarchical history" width="95%">
</p>

## Synchronized Gameplay and GUI Data

GameWAM training uses synchronized visual observations, native actions, state, and task information. The Minecraft data pipeline standardizes gameplay recordings into a shared keyboard–mouse action space, constructs event-anchored clips around behaviorally informative transitions, and generates scripted GUI interactions for crafting, smelting, and related inventory procedures. These sources are represented as aligned, WAM-ready trajectories with a common action layout.

<p align="center">
  <img src="assets/data-pipeline.webp" alt="GameWAM data pipeline" width="95%">
</p>

## Evaluation Scope

We evaluate GameWAM on native closed-loop control in **Minecraft** and **ViZDoom**. The two environments cover different visual dynamics, task horizons, and native action spaces. We additionally include a diagnostic **Minecraft-to-VoxeLibre zero-shot evaluation** to study transfer without VoxeLibre fine-tuning.

Across Minecraft and ViZDoom, GameWAM achieves competitive closed-loop performance. In Minecraft, it uses fewer executed native actions than the compared agents across all evaluated task categories. Full benchmark tables, evaluation protocols, and released configurations will accompany the paper and code release.

## Low-Frequency Action Source Imprinting

Our closed-loop analysis reveals **Low-Frequency Action Source Imprinting (LASI)**. Under fixed conditioning, low-temporal-frequency components of the sampled action source can coherently steer coarse generated camera motion. Reusing the same source across replanning steps can accumulate this bias into persistent directional motion, including repeated in-place rotation. This behavior distinguishes LASI from ordinary sampling diversity and exposes a source-sensitivity failure mode specific to repeated generative control.

## Citation

If you find GameWAM useful, please cite:

```bibtex
@article{guo2026gamewam,
  title         = {GameWAM: A World Action Model for Video Games},
  author        = {Guo, Yuncheng and Zhang, Zhanqiu and Guo, Yiwen and Li, Weijia},
  journal       = {arXiv preprint arXiv:2608.26200},
  year          = {2026},
  eprint        = {2608.26200},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.26200}
}
```
