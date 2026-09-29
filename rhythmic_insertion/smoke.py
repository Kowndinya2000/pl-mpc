"""Short GPU reset, step and optional rendering check."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--size", type=int, choices=(1, 3, 5), default=5)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    from rhythmic_insertion.env import make_env
    import numpy as np
    env = make_env({"task": "ri_insert", "device": args.device, "seed": 1,
                    "ri_asset_size": args.size, "ri_enable_camera": args.render})
    try:
        obs, _ = env.reset(seed=1)
        assert obs.shape == (17,) and np.isfinite(obs).all()
        for _ in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(np.zeros(6, dtype=np.float32))
            assert np.isfinite(obs).all() and 0 <= reward <= 1
            assert "success" in info and not terminated
            if truncated:
                obs, _ = env.reset()
        if args.render:
            frame = env.render()
            assert frame.shape == (256, 256, 3) and frame.std() > 0
        obs, _ = env.reset(seed=2)
        assert np.isfinite(obs).all()
        print(f"Insertion smoke passed: size={args.size}, steps={args.steps}, obs=17, actions=6.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
