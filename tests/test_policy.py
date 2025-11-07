import unittest

from dqar import DQARPolicy, PolicyConfig
from dqar.policy import PolicyFeatures


class PolicyTest(unittest.TestCase):
    def test_probability_bounds(self):
        policy = DQARPolicy(PolicyConfig(hidden_dim=4, num_hidden_layers=1))
        features = PolicyFeatures(
            entropy=0.5,
            snr=1.0,
            latent_norm=2.0,
            step_index=3,
            total_steps=10,
            prompt_length=16,
        )
        prob = policy.predict_proba(features)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_training_moves_probability(self):
        policy = DQARPolicy(PolicyConfig(hidden_dim=4, num_hidden_layers=1))
        features = PolicyFeatures(
            entropy=0.1,
            snr=10.0,
            latent_norm=1.0,
            step_index=8,
            total_steps=10,
            prompt_length=8,
        )
        initial = policy.predict_proba(features)
        dataset = [(
            features,
            1.0,
        )]
        policy.fit(dataset, epochs=5, lr=0.05)
        trained = policy.predict_proba(features)
        self.assertGreaterEqual(trained, initial)


if __name__ == "__main__":
    unittest.main()
