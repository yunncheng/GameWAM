"""Execute Minecraft commands after environment reset."""
from minestudio.simulator.callbacks.callback import MinecraftCallback

# Freeze world evolution while chunks settle, then restore requested values.
_SETTLE_FREEZE_GAMERULES = {
    "doDaylightCycle": ("false", "true"),
    "doWeatherCycle": ("false", "true"),
    "doMobSpawning": ("false", "true"),
    "doFireTick": ("false", "true"),
    "randomTickSpeed": ("0", "3"),
}


def _command_text(command) -> str:
    """Normalize a command entry to text (configs sometimes nest lists)."""
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)


class CommandsCallback(MinecraftCallback):

    def __init__(self, commands):
        super().__init__()
        self.commands = commands

    def after_reset(self, sim, obs, info):
        if not self.commands:
            return obs, info

        desired = {
            rule: default
            for rule, (_, default) in _SETTLE_FREEZE_GAMERULES.items()
        }
        time_commands = []
        for command in self.commands:
            parts = _command_text(command).strip().split()
            head = parts[0].lstrip("/") if parts else ""
            if len(parts) >= 3 and head == "gamerule" and parts[1] in desired:
                desired[parts[1]] = parts[2]
            elif len(parts) >= 3 and head == "time" and parts[1] == "set":
                time_commands.append(command)

        # Keep the player alive while the teleported region finishes loading.
        _POST_TP_SETTLE = 100
        _TOTAL_SETTLE = 600
        settled = 0
        for command in self.commands:
            obs, reward, done, info = sim.env.execute_cmd(command)
            parts = _command_text(command).strip().split()
            head = parts[0].lstrip("/") if parts else ""
            if head == "tp":
                sim.env.execute_cmd("/effect give @s water_breathing 999999 0 true")
                sim.env.execute_cmd("/effect give @s resistance 999999 255 true")
                settled += 2
                for _ in range(_POST_TP_SETTLE):
                    action = sim.env.noop_action()
                    obs, reward, done, info = sim.env.step(action)
                settled += _POST_TP_SETTLE

        for rule, (freeze, _) in _SETTLE_FREEZE_GAMERULES.items():
            sim.env.execute_cmd(f"/gamerule {rule} {freeze}")

        # A fixed budget keeps initial terrain rendering deterministic.
        for _ in range(_TOTAL_SETTLE - settled):
            action = sim.env.noop_action()
            obs, reward, done, info = sim.env.step(action)

        for rule, value in desired.items():
            sim.env.execute_cmd(f"/gamerule {rule} {value}")
        for command in time_commands:
            sim.env.execute_cmd(command)
        sim.env.execute_cmd("/effect clear @s water_breathing")
        sim.env.execute_cmd("/effect clear @s resistance")
        obs, info = sim._wrap_obs_info(obs, info)
        return obs, info
