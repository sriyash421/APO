import tensorflow as tf

from tonic import logger
from tonic.tensorflow import agents, updaters, models, normalizers
from segments import Segment

def default_model(actor_sizes=(64, 64), actor_activation='tanh',
                  critic_sizes=(64, 64), critic_activation='tanh',
                  observation_normalizer=None):
    return models.ActorCritic(
        actor=models.Actor(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(actor_sizes, actor_activation),
            head=models.DetachedScaleGaussianPolicyHead()),
        critic=models.Critic(
            encoder=models.ObservationEncoder(),
            torso=models.MLP(critic_sizes, critic_activation),
            head=models.ValueHead()),
        observation_normalizer=observation_normalizer)


class APO(agents.A2C):
    '''Average-Reward Reinforcement Learning with Trust Region Methods.
    APO: https://arxiv.org/pdf/2106.03442.pdf
    '''

    def __init__(
        self, alpha=0.1, v=0.1, model=None, replay=None, actor_updater=None, critic_updater=None
    ):
        actor_updater = actor_updater or updaters.ClippedRatio()
        model = model or default_model()
        replay = replay or Segment(discount_factor=1.0)
        
        self.alpha = alpha
        self.v = v
        
        super().__init__(
            model=model, replay=replay, actor_updater=actor_updater,
            critic_updater=critic_updater)

    def initialize(self, observation_space, action_space, seed=None):
        super().initialize(observation_space, action_space, seed=seed)
        self.model.initialize(observation_space, action_space)
        self.replay.initialize(seed)
        self.actor_updater.initialize(self.model)
        self.critic_updater.initialize(self.model)

        self.eta = 0
        self.b = 0

    def _update(self):
        # Compute the lambda-returns.
        batch = self.replay.get_full('observations', 'next_observations')
        values, next_values = self._evaluate(**batch)
        values, next_values = values.numpy(), next_values.numpy()
        
        rewards = self.replay.get_full('rewards')['rewards']
        self.eta = (1 - self.alpha) * self.eta + self.alpha * rewards.mean()
        self.b = (1 - self.alpha) * self.b + self.alpha * values.mean()

        self.replay.compute_returns(values, next_values, self.eta)

        train_actor = True
        actor_iterations = 0
        critic_iterations = 0
        keys = 'observations', 'actions', 'advantages', 'log_probs', 'returns'

        # Update both the actor and the critic multiple times.
        for batch in self.replay.get(*keys):
            if train_actor:
                infos = self._update_actor_critic(**batch)
                actor_iterations += 1
            else:
                batch = {k: batch[k] for k in ('observations', 'returns')}
                infos = dict(critic=self.critic_updater(**batch))
            critic_iterations += 1

            # Stop earlier the training of the actor.
            if train_actor:
                train_actor = not infos['actor']['stop'].numpy()

            for key in infos:
                for k, v in infos[key].items():
                    logger.store(key + '/' + k, v.numpy())

        logger.store('actor/iterations', actor_iterations)
        logger.store('critic/iterations', critic_iterations)
        logger.store('average_values', values.mean())

        # Update the normalizers.
        if self.model.observation_normalizer:
            self.model.observation_normalizer.update()
        if self.model.return_normalizer:
            self.model.return_normalizer.update()

    @tf.function
    def _update_actor_critic(
        self, observations, actions, advantages, log_probs, returns
    ):
        actor_infos = self.actor_updater(
            observations, actions, advantages, log_probs)
        
        # Average Value Constraint
        returns = returns + self.v * self.b

        critic_infos = self.critic_updater(observations, returns)
        return dict(actor=actor_infos, critic=critic_infos)
