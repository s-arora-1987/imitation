import numpy as np

from imitation.data import rollout
from imitation.policies.base import RandomPolicy
from imitation.util import reward_wrapper, util


class FunkyReward:
    def __call__(self, obs, act, next_obs, steps=None):
        # give each environment number from 1 to num_envs
        return (np.arange(len(obs)) + 1).astype("float32")


def test_reward_overwrite():
    """Test that reward wrapper actually overwrites base rewards."""
    env_name = "Pendulum-v0"
    num_envs = 3
    env = util.make_vec_env(env_name, num_envs)
    reward_fn = FunkyReward()
    wrapped_env = reward_wrapper.RewardVecEnvWrapper(env, reward_fn)
    policy = RandomPolicy(env.observation_space, env.action_space)
    sample_until = rollout.make_min_episodes(10)
    default_stats = rollout.rollout_stats(
        rollout.generate_trajectories(policy, env, sample_until)
    )
    wrapped_stats = rollout.rollout_stats(
        rollout.generate_trajectories(policy, wrapped_env, sample_until)
    )
    # Pendulum-v0 always has negative rewards
    assert default_stats["return_max"] < 0
    # ours gives between 1 * traj_len and num_envs * traj_len reward
    # (trajectories are all constant length of 200 in Pendulum)
    steps = wrapped_stats["len_mean"]
    assert wrapped_stats["return_min"] == 1 * steps
    assert wrapped_stats["return_max"] == num_envs * steps

    # check that wrapped reward is negative (all pendulum rewards is negative)
    # and other rewards are non-negative
    rand_act, _ = policy.predict(wrapped_env.reset())
    _, rew, _, infos = wrapped_env.step(rand_act)
    assert np.all(rew >= 0)
    assert np.all([info_dict["wrapped_env_rew"] < 0 for info_dict in infos])
