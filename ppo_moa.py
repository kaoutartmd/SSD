from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils import try_import_tf

from ray.rllib.policy.tf_policy_template import build_tf_policy
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.tf_policy import EntropyCoeffSchedule, LearningRateSchedule

from common_funcs_moa import (
    EXTRINSIC_REWARD,
    SOCIAL_INFLUENCE_REWARD,
    MOAResetConfigMixin,
    build_model,
    get_moa_mixins,
    moa_fetches,
    moa_postprocess_trajectory,
    setup_moa_loss,
    setup_moa_mixins,
    validate_moa_config,
)

from ray.rllib.algorithms.ppo.ppo_tf_policy import (
    KLCoeffMixin,
    PPOLoss,
    ValueNetworkMixin,
    clip_gradients,
    kl_and_loss_stats,
    postprocess_ppo_gae,
    setup_config,
    vf_preds_fetches,
)

tf = try_import_tf()

POLICY_SCOPE = "func"

def loss_with_moa(policy, model, dist_class, train_batch):
    logits, state = model.from_batch(train_batch)
    action_dist = dist_class(logits, model)

    moa_loss = setup_moa_loss(logits, policy, train_batch)
    policy.moa_loss = moa_loss.total_loss

    if state:
        max_seq_len = tf.reduce_max(train_batch["seq_lens"])
        mask = tf.sequence_mask(train_batch["seq_lens"], max_seq_len)
        mask = tf.reshape(mask, [-1])
    else:
        mask = tf.ones_like(train_batch[Postprocessing.ADVANTAGES], dtype=tf.bool)

    policy.loss_obj = PPOLoss(
        dist_class,
        model,
        train_batch[Postprocessing.VALUE_TARGETS],
        train_batch[Postprocessing.ADVANTAGES],
        train_batch[SampleBatch.ACTIONS],
        train_batch[SampleBatch.ACTION_DIST_INPUTS],
        train_batch[SampleBatch.ACTION_LOGP],
        train_batch[SampleBatch.VF_PREDS],
        action_dist,
        model.value_function(),
        policy.kl_coeff,
        mask,
        policy.entropy_coeff,
        policy.config["clip_param"],
        policy.config["vf_clip_param"],
        policy.config["vf_loss_coeff"],
        policy.config["use_gae"],
    )

    policy.loss_obj.loss += moa_loss.total_loss
    return policy.loss_obj.loss

def extra_moa_fetches(policy):
    ppo_fetches = vf_preds_fetches(policy)
    ppo_fetches.update(moa_fetches(policy))
    return ppo_fetches

def extra_moa_stats(policy, train_batch):
    base_stats = kl_and_loss_stats(policy, train_batch)
    base_stats.update({
        "var_gnorm": tf.global_norm([x for x in policy.model.trainable_variables()]),
        "cur_influence_reward_weight": tf.cast(policy.cur_influence_reward_weight_tensor, tf.float32),
        SOCIAL_INFLUENCE_REWARD: train_batch[SOCIAL_INFLUENCE_REWARD],
        EXTRINSIC_REWARD: train_batch[EXTRINSIC_REWARD],
        "moa_loss": policy.moa_loss,
    })
    return base_stats

def postprocess_ppo_moa(policy, sample_batch, other_agent_batches=None, episode=None):
    batch = moa_postprocess_trajectory(policy, sample_batch)
    batch = postprocess_ppo_gae(policy, batch)
    return batch

def setup_ppo_moa_mixins(policy, obs_space, action_space, config):
    ValueNetworkMixin.__init__(policy, obs_space, action_space, config)
    KLCoeffMixin.__init__(policy, config)
    EntropyCoeffSchedule.__init__(policy, config["entropy_coeff"], config["entropy_coeff_schedule"])
    LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
    setup_moa_mixins(policy, obs_space, action_space, config)

def validate_ppo_moa_config(config):
    validate_moa_config(config)

class MOAPPOTFPolicyClass:
    @staticmethod
    def policy_class(config):
        return build_tf_policy(
            name="MOAPPOTFPolicy",
            get_default_config=lambda: config,
            loss_fn=loss_with_moa,
            make_model=build_model,
            stats_fn=extra_moa_stats,
            extra_action_fetches_fn=extra_moa_fetches,
            postprocess_fn=postprocess_ppo_moa,
            gradients_fn=clip_gradients,
            before_init=setup_config,
            before_loss_init=setup_ppo_moa_mixins,
            mixins=[
                LearningRateSchedule,
                EntropyCoeffSchedule,
                KLCoeffMixin,
                ValueNetworkMixin
            ] + get_moa_mixins(),
        )

def build_ppo_moa_trainer(moa_config_dict):
    validate_ppo_moa_config(moa_config_dict)

    config = PPOConfig().update_from_dict(moa_config_dict)
    config = config.multi_agent(
        policies={
            "default_policy": PolicySpec(policy_class=MOAPPOTFPolicyClass.policy_class)
        },
        policy_mapping_fn=lambda agent_id, *args, **kwargs: "default_policy"
    )

    return config.build()
