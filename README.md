# Reinforcement Learning Notes

这是一个强化学习学习仓库，主要跟着 [动手学强化学习](https://hrl.boyuai.com/) 做练习，并把代码和笔记整理在这里。

## 项目结构

- `basic/`：基础内容，来自上述教材
- `isaaclab_ground_ball/`：Isaac Lab 相关实验

## 环境配置

推荐使用 `conda`：

```bash
conda create -n rl-study python=3.12 -y
conda activate rl-study
python3 -m pip install -r requirements.txt
```

## 已知问题

目前有一个依赖相关问题，临时处理方式如下：

```bash
echo 'export LD_LIBRARY_PATH=/home/yuqingchi/anaconda3/envs/legged-gym-dev/lib:$LD_LIBRARY_PATH' >> ~/.zshrc
source ~/.zshrc
```
