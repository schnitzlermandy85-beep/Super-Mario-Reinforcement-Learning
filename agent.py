"""
学生作业文件：在马里奥场景中实现 DQN。

你需要补全的核心函数：
- `td_estimate`
- `td_target`
- `update_Q_online`
"""

from collections import deque
from pathlib import Path
import random
import shutil

import numpy as np
import torch

from neural import MarioNet

import warnings
warnings.filterwarnings("ignore", message="Gym has been unmaintained")


# ── 兼容性补丁（只执行一次）──────────────────────────────────────────────────
if "agent_patched" not in dir():
    agent_patched = True

    import sys as _sys
    from pathlib import Path as _Path

    # 1. 拷贝 ROM 到纯 ASCII 路径，绕过 Cython 扩展的 fopen 编码问题
    _rom_dir = _Path(r"C:\Users\Public\.mario_roms")
    _rom_dir.mkdir(parents=True, exist_ok=True)
    _rom_file = _rom_dir / "super-mario-bros.nes"
    if not _rom_file.is_file():
        import gym_super_mario_bros._roms
        _src = _Path(gym_super_mario_bros._roms.rom_path(False, "vanilla"))
        if _src.is_file():
            shutil.copy2(str(_src), str(_rom_file))

    # 2. 替换 rom_path() 使其返回 ASCII 路径
    import gym_super_mario_bros

    # 保存原始函数引用，避免递归
    _orig_rom_path = gym_super_mario_bros._roms.rom_path

    def _rom_path(lost_levels, rom_mode):
        name = _Path(_orig_rom_path(lost_levels, rom_mode)).name
        local = _rom_dir / name
        return str(local) if local.is_file() else _orig_rom_path(lost_levels, rom_mode)

    # 更新子模块 + 包级引用
    _sys.modules["gym_super_mario_bros._roms.rom_path"].rom_path = _rom_path
    gym_super_mario_bros._roms.rom_path = _rom_path

    # 3. 替换 wrappers.make_env，绕过 gym/gymnasium 混用
    import wrappers as _wrappers
    from compat import patch_runtime_compat

    # 用 gymnasium 重新实现 wrappers.py 中的包装器，维持 gymnasium.Env 类型链
    import gymnasium as _gym

    class _SkipFrame(_gym.Wrapper):
        def __init__(self, env, skip):
            super().__init__(env)
            self._skip = skip
        def step(self, action):
            total_reward = 0.0
            terminated = truncated = False
            info = {}
            obs = None
            for _ in range(self._skip):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            return obs, total_reward, terminated, truncated, info

    class _GrayScaleObservation(_gym.ObservationWrapper):
        def __init__(self, env, keep_dim=False):
            super().__init__(env)
            self.keep_dim = keep_dim
            obs_shape = self.observation_space.shape[:2]
            if keep_dim:
                obs_shape = obs_shape + (1,)
            self.observation_space = _gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8,
            )
        def observation(self, observation):
            observation = observation.astype(np.float32, copy=False)
            gray = (0.299 * observation[..., 0] + 0.587 * observation[..., 1] + 0.114 * observation[..., 2]).astype(np.uint8)
            if self.keep_dim:
                gray = np.expand_dims(gray, axis=-1)
            return gray

    class _ResizeObservation(_gym.ObservationWrapper):
        def __init__(self, env, shape):
            super().__init__(env)
            if isinstance(shape, int):
                self.shape = (shape, shape)
            else:
                self.shape = tuple(shape)
            input_shape = self.observation_space.shape
            self._src_height = input_shape[0]
            self._src_width = input_shape[1]
            self._row_idx = np.linspace(0, self._src_height - 1, self.shape[0]).astype(np.int32)
            self._col_idx = np.linspace(0, self._src_width - 1, self.shape[1]).astype(np.int32)
            obs_shape = self.shape + self.observation_space.shape[2:]
            self.observation_space = _gym.spaces.Box(
                low=0, high=255, shape=obs_shape, dtype=np.uint8,
            )
        def observation(self, observation):
            return observation[self._row_idx][:, self._col_idx]

    class _NormalizeObservation(_gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = _gym.spaces.Box(
                low=0.0, high=1.0, shape=self.observation_space.shape, dtype=np.float32,
            )
        def observation(self, observation):
            return observation.astype(np.float32, copy=False) / 255.0

    def _make_env(level="SuperMarioBros-1-1-v0"):
        patch_runtime_compat()
        from gym_super_mario_bros.smb_env import SuperMarioBrosEnv

        # smb_env 在 import 时绑定了本地 rom_path 引用，这里也打上补丁
        _sys.modules["gym_super_mario_bros.smb_env"].rom_path = _rom_path

        _MODES = ["vanilla", "downsample", "pixel", "rectangle"]
        _parts = level.split("-")
        if len(_parts) == 4 and _parts[1].isdigit() and _parts[2].isdigit():
            _ver = int(_parts[3][1:])
            _env = SuperMarioBrosEnv(
                rom_mode=_MODES[_ver],
                target=(int(_parts[1]), int(_parts[2])),
            )
        else:
            _ver = int(level.rpartition("v")[-1])
            _env = SuperMarioBrosEnv(rom_mode=_MODES[_ver])

        from nes_py.wrappers import JoypadSpace

        _env = JoypadSpace(_env, [["right"], ["right", "A"]])
        _env = _SkipFrame(_env, skip=4)
        _env = _GrayScaleObservation(_env, keep_dim=False)
        _env = _ResizeObservation(_env, shape=84)
        _env = _NormalizeObservation(_env)
        from gymnasium.wrappers.stateful_observation import FrameStackObservation

        _env = FrameStackObservation(_env, stack_size=4)
        # 最外层：把 gymnasium 的 5 返回值 (obs,reward,terminated,truncated,info)
        # 转为旧式 4 返回值 (obs,reward,done,info) 适配 main.py
        class _StepCompat(_gym.Wrapper):
            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                return obs, reward, terminated or truncated, info
            def reset(self, **kwargs):
                obs, _ = self.env.reset(**kwargs)
                return obs

        _env = _StepCompat(_env)
        return _env

    _wrappers.make_env = _make_env
# ── 补丁结束 ──────────────────────────────────────────────────────────────


class Mario:
    def __init__(
        self,
        state_dim,
        action_dim,
        save_dir: Path,
        checkpoint=None,
        gpu_id=0,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir
        if gpu_id is not None and torch.cuda.is_available():
            if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
                raise ValueError(f"Invalid gpu_id={gpu_id}, available GPUs: {torch.cuda.device_count()}")
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
        self.use_cuda = self.device.type == "cuda"

        self.memory = deque(maxlen=100000)
        self.batch_size = 32

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.9999995
        self.exploration_rate_min = 0.1
        self.gamma = 0.9

        self.curr_step = 0
        self.burnin = 10000
        self.learn_every = 3
        self.sync_every = 10000
        self.save_every = 500000

        self.net = MarioNet(self.state_dim, self.action_dim).float()
        if self.use_cuda:
            self.net = self.net.to(device=self.device)
        if checkpoint is not None:
            self.load(checkpoint)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = torch.nn.SmoothL1Loss()

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim)
        else:
            state = np.asarray(state, dtype=np.float32)
            state = torch.as_tensor(state).unsqueeze(0)
            if self.use_cuda:
                state = state.to(self.device)
            action_values = self.net(state, model="online")
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        state = torch.as_tensor(np.asarray(state, dtype=np.float32))
        next_state = torch.as_tensor(np.asarray(next_state, dtype=np.float32))
        action = torch.tensor([action], dtype=torch.long)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.bool)

        # replay buffer 存 CPU 上，避免撑爆显存
        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))

        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()

    def td_estimate(self, state, action):
        """
        根据 online 网络返回当前 batch 的 Q(s, a)。

        提示：online 网络算 Q 值表，再按 action 取对应分数。见 ASSIGNMENT.md。
        """
        current_q_values = self.net(state, model="online")
        current_q = current_q_values[np.arange(0, self.batch_size), action]
        return current_q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """
        根据 Double DQN 目标公式计算 TD target，降低 Q 值高估。

        提示：online 网络选动作，target 网络估值。
        """
        # Double DQN: online 网络选最优动作
        next_actions = self.net(next_state, model="online").argmax(dim=1)
        # target 网络对该动作估值
        next_q = self.net(next_state, model="target").gather(1, next_actions.unsqueeze(1)).squeeze()
        return (reward + (1 - done.float()) * self.gamma * next_q).float()

    def update_Q_online(self, td_estimate, td_target):
        """
        使用 `self.loss_fn`、`self.optimizer` 完成一次参数更新。

        提示：loss_fn → zero_grad → backward → step → return loss.item()。
        """
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

    def learn(self):
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0 and self.curr_step > 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        state, next_state, action, reward, done = self.recall()
        if self.use_cuda:
            state = state.to(self.device)
            next_state = next_state.to(self.device)
            action = action.to(self.device)
            reward = reward.to(self.device)
            done = done.to(self.device)
        td_est = self.td_estimate(state, action)
        td_tgt = self.td_target(reward, next_state, done)
        loss = self.update_Q_online(td_est, td_tgt)
        return td_est.mean().item(), loss

    def save(self, save_name=None):
        if save_name is None:
            save_path = self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
        else:
            save_path = self.save_dir / save_name
        torch.save(
            {"model": self.net.state_dict(), "exploration_rate": self.exploration_rate},
            save_path,
        )
        print(f"Saved checkpoint to {save_path}")
        return save_path

    def load(self, load_path):
        checkpoint = torch.load(
            load_path,
            map_location=self.device,
            weights_only=True,
        )
        self.net.load_state_dict(checkpoint["model"])
        self.exploration_rate = checkpoint["exploration_rate"]
