import copy
import sys
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.registry import get_algorithm_class as get_agent_class
from ray.rllib.models import ModelCatalog
from ray.tune import Experiment
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from ppo_baseline import build_ppo_baseline_trainer
from ppo_moa import build_ppo_moa_trainer  # Use relative import if ppo_moa.py is in the same directory
from baseline_model import BaselineModel
from moa_model import MOAModel
from social_dilemmas.envs.env_creator import get_env_creator

model = "moa"
num_agents = 2
use_collective_reward = True
tune_hparams = False

def update_nested_dict(d0, d1):
    """
    Recursively updates a nested dictionary with a second nested dictionary.
    """
    for k, v in d1.items():
        if k in d0 and type(v) is dict:
            if type(d0[k]) is dict:
                update_nested_dict(d0[k], d1[k])
            else:
                raise TypeError
        else:
            d0[k] = d1[k]

def build_experiment_config_dict():
    """
    Create a config dict for a single Experiment object.
    """
    env_creator = get_env_creator(use_collective_reward)
    env_name = "cleanup_env"
    register_env(env_name, env_creator)

    single_env = env_creator(num_agents)
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    model_name = model + "_lstm"
    if model == "moa":
        ModelCatalog.register_custom_model(model_name, MOAModel)
    elif model == "baseline":
        ModelCatalog.register_custom_model(model_name, BaselineModel)

    def gen_policy():
        return None, obs_space, act_space, {"custom_model": model_name}

    # Create policy for each agent
    policy_graphs = {}
    for i in range(num_agents):
        policy_graphs["agent-" + str(i)] = gen_policy()

    def policy_mapping_fn(agent_id):
        return agent_id

    agent_cls = get_agent_class("PPO")
    config = copy.deepcopy(agent_cls._default_config)

    config["env"] = env_name

    config["env_config"]["func_create"] = env_creator
    config["env_config"]["env_name"] = env_name

    conv_filters = [[6, [3, 3], 1]]
    fcnet_hiddens = [32, 32]
    lstm_cell_size = 128

    train_batch_size = max(1, 4) * 8 * 1000          # 4 workers, 8 envs per worker, 1000 samples taken from workers

    update_nested_dict(
        config,
        {
            "horizon": 1000,
            "gamma": 0.99,
            "lr": 0.0001,
            "lr_schedule": None,
            "rollout_fragment_length": 1000,
            "train_batch_size": train_batch_size,
            "num_workers": 4,
            "num_envs_per_worker": 8,
            "num_gpus": 1,
            "num_cpus_for_driver": 0,
            "num_gpus_per_worker": 0,
            "num_cpus_per_worker": 1,
            "entropy_coeff": 0.001,
            "grad_clip": 40,
            "multiagent": {"policies": policy_graphs, "policy_mapping_fn": policy_mapping_fn},
            "callbacks": single_env.get_environment_callbacks(),
            "model": {
                "custom_model": model_name,
                "use_lstm": False,
                "conv_filters": conv_filters,
                "fcnet_hiddens": fcnet_hiddens,
                "custom_options": {
                    "cell_size": lstm_cell_size,
                    "num_other_agents": num_agents - 1,
                },
            },
        },
    )

    if tune_hparams:
        tune_dict = create_hparam_tune_dict(model=model, is_config=True)
        update_nested_dict(config, tune_dict)

    config.update(
        {
            "num_sgd_iter": 10,
            "sgd_minibatch_size": train_batch_size / 4,
            "vf_loss_coeff": 1e-4,
            "vf_share_layers": True,
        }
    )

    return config


def get_trainer(config):
    """
    Creates a trainer depending on what args are specified.
    """
    if model == "baseline":
        trainer = build_ppo_baseline_trainer(config)
    elif model == "moa":
        trainer = build_ppo_moa_trainer(config)
    if trainer is None:
        raise NotImplementedError("The provided combination of model and algorithm was not found.")
    return trainer


def initialize_ray():
    """
    Initialize ray and automatically turn on local mode when debugging.
    """
    ichanged_args_local_mode = False
    if sys.gettrace() is not None:
        print(
            "Debug mode detected through sys.gettrace(), turning on ray local mode. Saving"
            " experiment under ray_results/debug_experiment"
        )
        ichanged_args_local_mode = True
    ray.init(
        address=None,
        local_mode=ichanged_args_local_mode,
        include_webui=False,
    )

def build_experiment_dict(experiment_name, trainer, config):
    """
    Creates all parameters needed to create an Experiment object and puts them into a dict.
    """
    experiment_dict = {
        "name": experiment_name,
        "run": trainer,
        "stop": {},
        "checkpoint_freq": 100,
        "config": config,
        "num_samples": 1,
        "max_failures": -1,
    }

    experiment_dict["stop"]["timesteps_total"] = 5e6

    return experiment_dict


def create_experiment():
    experiment_name = "cleanup_" + model + "_PPO"
    config = build_experiment_config_dict()
    trainer = get_trainer(config=config)
    experiment_dict = build_experiment_dict(experiment_name, trainer, config)
    return Experiment(**experiment_dict)


def create_hparam_tune_dict(model, is_config=False):
    def wrapper(fn):
        if is_config:
            return tune.sample_from(lambda spec: fn)
        else:
            return lambda: fn

    baseline_options = {}
    model_options = {}
    if model == "baseline":
        baseline_options = {
            "entropy_coeff": wrapper(np.random.exponential(1 / 1000)),
            "lr": wrapper(np.random.uniform(0.00001, 0.01)),
        }
    if model == "moa":
        model_options = {
            "moa_loss_weight": wrapper(np.random.exponential(1 / 15)),
            "influence_reward_weight": wrapper(np.random.exponential(1)),
        }
    hparam_dict = {
        **baseline_options,
        "model": {"custom_options": model_options},
    }
    return hparam_dict


def create_pbt_scheduler(model):
    hyperparam_mutations = create_hparam_tune_dict(model=model, is_config=False)

    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=10,
        metric="episode_reward_mean",
        mode="max",
        hyperparam_mutations=hyperparam_mutations,
    )
    return pbt


def run(experiments):
    initialize_ray()
    scheduler = create_pbt_scheduler(model) if tune_hparams else None
    tune.run_experiments(
        experiments,
        queue_trials=False,
        scheduler=scheduler,
        reuse_actors=tune_hparams,
    )


if __name__ == "__main__":
    experiment = create_experiment()
    run(experiment)
