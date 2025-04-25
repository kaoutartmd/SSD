import gym
import numpy as np
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

# Step 1: Define a custom model
class CustomTFModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(CustomTFModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(num_outputs)
        ])
        self._value_branch = tf.keras.layers.Dense(1)
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        self._last_obs = input_dict["obs"]
        model_out = self.model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        assert self._last_obs is not None, "must call forward() first"
        return tf.reshape(self._value_branch(self._last_obs), [-1])

# Step 2: Register the custom model
ModelCatalog.register_custom_model("custom_tf_model", CustomTFModel)

# Step 3: Configure PPO to use the custom model
config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("tf")
    .training(model={"custom_model": "custom_tf_model"})
)

# Build the trainer
trainer = config.build()

# Train the agent
for i in range(3):
    result = trainer.train()
    print(f"Iteration {i}: episode_reward_mean = {result['episode_reward_mean']}")

# Clean up
trainer.cleanup()
