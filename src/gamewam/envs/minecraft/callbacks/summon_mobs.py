"""Summon mobs callback. Ported from OpenHA openagents/envs/callbacks/summon_mobs.py

Summons specified mobs around the player after environment reset.
"""
import numpy as np
from minestudio.simulator.callbacks.callback import MinecraftCallback


class SummonMobsCallback(MinecraftCallback):

    _SETTLE_STEPS = 10

    def __init__(self, mobs, *, seed: int):
        """
        Args:
            mobs: List of mob configs, e.g.:
                [{'name': 'cow', 'number': 10, 'range_x': [-5, 5], 'range_z': [-5, 5]}]
            seed: Seed for reproducible mob positioning.
        """
        self.mobs = mobs
        self.rng = np.random.default_rng(int(seed))

    def after_reset(self, sim, obs, info):
        chats = []
        for mob in self.mobs:
            for _ in range(mob['number']):
                name = mob['name']
                x = self.rng.uniform(*mob['range_x'])
                z = self.rng.uniform(*mob['range_z'])
                chat = f'/execute as @p at @p run summon minecraft:{name} ~{x} ~ ~{z} {{Age:0}}'
                chats.append(chat)
        for chat in chats:
            obs, reward, done, info = sim.env.execute_cmd(chat)
        if chats:
            for _ in range(self._SETTLE_STEPS):
                action = sim.env.noop_action()
                obs, reward, done, info = sim.env.step(action)
            obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
