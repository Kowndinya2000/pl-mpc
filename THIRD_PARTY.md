# Third-party software and assets

Required copyright notices are retained. PL-MPC author attribution does not replace upstream attribution.

| Component | Origin and notice |
| --- | --- |
| TD-M(PC)² backbone and separate baseline | [Upstream repository](https://github.com/DarthUtopian/tdmpc_square_public); MIT, copyright Haotian Lin. See `LICENSE` and `licenses/TD-MPC-Square-LICENSE`. |
| TD-MPC2-derived components | [Upstream repository](https://github.com/nicklashansen/tdmpc2); MIT, copyright Nicklas Hansen. See `licenses/TD-MPC2-LICENSE`. |
| DrQ-v2-derived image augmentation and DMControl wrappers | [Upstream repository](https://github.com/facebookresearch/drqv2); MIT, copyright Facebook, Inc. and its affiliates. See `licenses/DrQ-v2-LICENSE`. |
| DreamerV3-derived symmetric log/exp transforms | [Upstream repository](https://github.com/danijar/dreamerv3); MIT, copyright Danijar Hafner. The notice is included in the DreamerV3 section of `licenses/HumanoidBench-LICENSE`. |
| HumanoidBench code and selected H1Hand assets | [Upstream repository](https://github.com/carlosferrazza/humanoid-bench); see the complete upstream `licenses/HumanoidBench-LICENSE` and retained notices in `humanoid_bench/assets/h1/` and `shadow_hand_menagerie/`. Only the dependency closure for the 13 bundled tasks is distributed. |
| dm_control | [Upstream repository](https://github.com/google-deepmind/dm_control); Apache 2.0, see `licenses/dm_control-LICENSE`. Installed as a dependency. |
| Gym-derived time-limit wrapper | Source revision is cited in the wrapper. MIT notice is retained in `licenses/Gym-LICENSE`. The wrapper uses the Gymnasium API. |

Third-party simulator and framework dependencies retain their own licenses in the installed packages.

The rhythmic insertion simulator derives from the wrench–nut benchmark in *Failure Forecasting Boosts Robustness of Sim2Real Rhythmic Insertion Policies* and NVIDIA IsaacGymEnvs/IndustReal. The retained NVIDIA BSD-3-Clause notice is in `licenses/IsaacGymEnvs-LICENSE` and the source headers. The bundled KUKA/Robotiq and nut–bolt–wrench descriptions and meshes come from the benchmark's simulator tree. The proprietary NVIDIA Isaac Gym Preview 4 runtime is installed separately and is not part of this archive.
