from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn a ground plane and a small ball in Isaac Lab.")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of simulation steps to run. Use 0 to run until the app is closed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils


def design_scene() -> None:
    """Create a minimal scene with a ground plane and a ball."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # A single light keeps the scene visible in the interactive viewer.
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/DomeLight", light_cfg)

    ball_cfg = sim_utils.SphereCfg(
        radius=0.1,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.2),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2), metallic=0.0),
    )
    ball_cfg.func("/World/Ball", ball_cfg, translation=(0.0, 0.0, 1.0))


def main() -> None:
    """Start the simulator and step the minimal scene."""
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([2.0, 2.0, 1.5], [0.0, 0.0, 0.2])

    design_scene()

    sim.reset()
    print("[INFO]: Minimal scene is ready.")

    step_count = 0
    while simulation_app.is_running():
        sim.step()
        step_count += 1
        if args_cli.steps > 0 and step_count >= args_cli.steps:
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
