# Isaac Lab Ground Ball

Minimal Isaac Lab project in this workspace.

The scene contains:

- one ground plane
- one rigid ball

## Run

Launch the interactive app:

```bash
cd /home/yuqingchi/Code/RL/isaaclab_ground_ball
conda run -n isaaclab python scripts/ground_ball.py
```

Run a short headless check:

```bash
cd /home/yuqingchi/Code/RL/isaaclab_ground_ball
conda run -n isaaclab python scripts/ground_ball.py --headless --steps 120
```

## Notes

- `--steps 0` means keep running until the app is closed.
- The script uses the existing local conda environment: `isaaclab`.
