#!/usr/bin/env python3
"""
学生版训练入口：DQN。
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

from agent import Mario
from metrics import MetricLogger
from wrappers import make_env


def main():
    parser = argparse.ArgumentParser(description="Train Mario with DQN")
    parser.add_argument("--level", default="SuperMarioBros-1-1-v0")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    save_dir = args.save_dir or (
        Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(args.level)
    mario = Mario(
        state_dim=(4, 84, 84),
        action_dim=env.action_space.n,
        save_dir=save_dir,
        gpu_id=args.gpu,
    )
    logger = MetricLogger(save_dir)

    for e in range(args.episodes):
        state = env.reset()

        while True:
            action = mario.act(state)
            next_state, reward, done, info = env.step(action)

            mario.cache(state, next_state, action, reward, done)
            q, loss = mario.learn()
            logger.log_step(reward, loss, q)

            state = next_state
            if done or info.get("flag_get", False):
                break

        logger.log_episode()
        if e % 20 == 0 or e == args.episodes - 1:
            logger.record(
                episode=e,
                epsilon=mario.exploration_rate,
                step=mario.curr_step,
            )

    if mario.curr_step > 0:
        mario.save("mario_net_final.chkpt")

    env.close()


if __name__ == "__main__":
    main()
