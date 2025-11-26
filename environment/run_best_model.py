"""
run_best_model.py
Loads a saved model (if provided) and runs it in the FireRescueEnv.
Keeps the PyBullet GUI alive and forces extra fire updates so particles are visible.
Usage:
    python run_best_model.py            # runs in test (random) mode if no model found
    python run_best_model.py --model models/best_model.zip
"""

import argparse
import time
import os
import numpy as np

# Adjust import path/name if your env file is named differently
from custom_env import FireRescueEnv

# Optional: import stable-baselines if you will load a model
try:
    from stable_baselines3 import PPO, A2C, DQN
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


def safe_action_from_model(action, action_space):
    """
    Normalize model output to a valid discrete action (int) or continuous list.
    This handles numpy arrays returned by SB3.
    """
    # If numpy array, try to convert to scalar or list
    if isinstance(action, np.ndarray):
        # Common case: shape (1,) for discrete
        if action.size == 1:
            return int(action.item())
        # If multiple values but action space is discrete, take first element
        if getattr(action_space, "n", None) is not None:
            return int(action.flatten()[0])
        # Otherwise return a python list for continuous actions
        return action.astype(float).tolist()

    # If it's a list/tuple, cast according to action space
    if isinstance(action, (list, tuple)):
        if getattr(action_space, "n", None) is not None:
            # discrete -> take first element as int
            return int(action[0])
        return [float(x) for x in action]

    # If already an int or float, leave it
    if isinstance(action, (int, np.integer)):
        return int(action)
    if isinstance(action, (float, np.floating)):
        return float(action)

    # Fallback: return 0 (no-op)
    return 0


def force_fire_extra_updates(env, n=2, upward_drift=0.001):
    """
    Extra nudges to fire particles to make them more visible.
    - Calls the renderer step additional times
    - Slightly moves existing visual particles upward to emphasize motion
    """
    # Call renderer n extra times (safe: renderer checks connections)
    for _ in range(n):
        try:
            if hasattr(env, "fire_renderer") and env.fire_renderer is not None:
                env.fire_renderer.step()
        except Exception:
            pass

    # Additionally nudge visual particles (if present)
    try:
        import pybullet as p
        if hasattr(env, "fire_renderer") and env.fire_renderer is not None:
            for ptl in list(env.fire_renderer.particles):
                try:
                    # ptl.multi holds the multi-body id in env implementation
                    if getattr(ptl, "multi", None) is not None:
                        pos, orn = p.getBasePositionAndOrientation(ptl.multi)
                        # small upward nudge
                        newpos = (pos[0], pos[1], pos[2] + upward_drift)
                        p.resetBasePositionAndOrientation(ptl.multi, newpos, orn)
                except Exception:
                    continue
    except Exception:
        # pybullet may not be importable here (should be), ignore safely
        pass


def load_sb3_model(model_path):
    """
    Try to infer the algorithm from zip metadata (best-effort).
    If SB3 isn't available, return None.
    """
    if not SB3_AVAILABLE:
        print("stable-baselines3 not available; running in random/test mode.")
        return None

    # Try common algos in priority order
    for Algo in (PPO, A2C, DQN):
        try:
            model = Algo.load(model_path)
            print(f"Loaded model as {Algo.__name__}")
            return model
        except Exception:
            continue

    # Last attempt: let user know
    print("Could not auto-detect model type or failed to load via PPO/A2C/DQN.")
    return None


def run_loop(model_path=None, episodes=100, render=True, keep_open=True):
    # Create environment
    env = FireRescueEnv(render_mode="human" if render else None, max_steps=500, headless=not render)
    print("Environment created.")

    model = None
    if model_path:
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}. Starting in test/random mode.")
            model_path = None
        else:
            model = load_sb3_model(model_path)

    obs, info = env.reset()
    episode = 0
    episode_reward = 0.0
    episode_steps = 0

    print("\n=== RUN STARTED ===")
    print("Press Ctrl+C to stop and close the GUI.\n")

    try:
        while True:
            # If model exists, predict; otherwise sample random action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            # Normalize action to python-friendly type
            action = safe_action_from_model(action, env.action_space)

            # Step env
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1

            # Force extra fire updates so particles remain visible
            # (calls renderer.step() extra times + nudges visuals)
            force_fire_extra_updates(env, n=2, upward_drift=0.002)

            # Small sleep for smooth GUI (approx 60 FPS)
            time.sleep(0.016)

            # Episode end handling
            if terminated or truncated:
                episode += 1
                print(f"Episode {episode:3d} | Reward: {episode_reward:8.2f} | Steps: {episode_steps:4d} | "
                      f"Terminated={terminated} Truncated={truncated}")
                # Reset counters and environment
                obs, info = env.reset()
                episode_reward = 0.0
                episode_steps = 0

                # If user set a finite episodes, stop when reached
                if episodes is not None and episodes > 0 and episode >= episodes and not keep_open:
                    print("Finished requested episodes.")
                    break

            # If user doesn't want to keep open, exit after episodes
            if episodes is not None and episodes > 0 and episode >= episodes and not keep_open:
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting...")

    finally:
        # Keep GUI alive if requested (so you can observe the final scene).
        if keep_open and p_is_connected_safe():
            print("Keeping GUI open. Press Ctrl+C again to close.")
            try:
                while True:
                    # Let fire keep animating
                    force_fire_extra_updates(env, n=1, upward_drift=0.0008)
                    time.sleep(0.25)
            except KeyboardInterrupt:
                print("\nClosing and cleaning up...")

        # Cleanly close environment
        try:
            env.close()
        except Exception:
            pass

        print("Done.")


def p_is_connected_safe():
    """Safe check for pybullet connection"""
    try:
        import pybullet as p
        return p.isConnected()
    except Exception:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default=None, help="Path to saved model (zip)")
    parser.add_argument("--episodes", "-e", type=int, default=0, help="Number of episodes to run (0 = infinite)")
    parser.add_argument("--no-render", action="store_true", help="Run in headless mode (no GUI)")
    parser.add_argument("--no-keep", action="store_true", help="Do not keep GUI open after finish")
    args = parser.parse_args()

    run_loop(
        model_path=args.model,
        episodes=None if args.episodes == 0 else args.episodes,
        render=not args.no_render,
        keep_open=not args.no_keep
    )

