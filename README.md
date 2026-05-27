# Mario DQN Starter Code

该框架目标是在 `gym-super-mario-bros` 上补全并跑通 DQN。

这是马里奥像素场景更常见、更合理的强化学习做法。

## 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate mario-dqn
```

## 2. File Structure

- `main.py`：训练入口
- `agent.py`：需要补全的 DQN 核心逻辑
- `neural.py`：CNN 网络定义
- `wrappers.py`：环境预处理
- `compat.py`：老版本依赖的运行时兼容补丁
- `metrics.py`：训练日志与曲线
- `replay.py`：加载 checkpoint 回放

## 3. What To Read

主要阅读：

- `agent.py`
- `neural.py`
- `wrappers.py`
- `main.py`

其中 `agent.py` 中已经标出 `TODO`，需要自行补全。

## 4. Run

```bash
python main.py --episodes 100 --gpu 0
```

这里的 `100` 只是用于快速检查代码是否能够正常运行的短跑示例。
如果有多张 GPU，可以将 `--gpu 0` 改成对应的卡号。

长周期正式训练示例：

```bash
python main.py --episodes 10000 --gpu 0
```

## 5. 长周期训练与收敛分析

马里奥这类像素控制任务，和作业说明（`ASSIGNMENT.md`）里的要求一致：**通常需要长周期训练**，不能只看几十个 episode 就下结论。

**为什么需要长周期？**

- 奖励稀疏：马里奥经常长时间拿不到明显正反馈，前期曲线波动大。
- 探索占比较高：epsilon 衰减较慢时，前几千个 episode 可能主要在“乱试”。
- 输入复杂：画面要经过预处理再进 CNN，策略从随机到稳定需要足够多的样本。

因此，想判断训练是否真的有效，建议至少跑到 **上万 episode（如 10000+）**，再结合整体趋势分析，而不是只看某一次短跑。

**怎么看是否收敛？**

训练过程中，`metrics.py` 会在 checkpoint 目录下保存日志和曲线（如 `log.txt`、`reward_plot.jpg`、`loss_plot.jpg`、`q_plot.jpg`）。建议重点看：

| 指标 | 大致含义 | 收敛时常见表现 |
|------|----------|----------------|
| reward（滑动平均） | 每局累计奖励 | 整体缓慢上升，后期波动变小或进入平台期 |
| loss | TD 误差 | 前期较大，后期趋于稳定（不必强求降到 0） |
| Q value | 网络估计的价值 | 随训练逐步变化，后期变化幅度变小 |

可以按阶段描述（与作业报告要求一致）：

1. **前期**：探索为主，reward 波动大，曲线起伏明显。
2. **中期**：若实现正确，平均 reward 可能缓慢抬升。
3. **后期**：观察是否进入平台期、继续缓慢提升，或长期不稳定——并在报告中说明你的判断。

**作业拔高部分（10 分）** 即考察上述长周期训练与收敛分析，口径与 `ASSIGNMENT.md` 完全一致。

## 6. Replay A Checkpoint

```bash
python replay.py --checkpoint checkpoints/your_run/mario_net_final.chkpt --gpu 0
```

## 7. Notes

- 重点是理解 DQN 的训练流程，不要求必须通关
