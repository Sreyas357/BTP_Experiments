from mdp import MDP


def build_small_mdp() -> MDP:
	return MDP(
		states=["s1", "s2", "s3"],
		actions=["a"],
		state_action_map={"s1": ["a"], "s2": ["a"], "s3": ["a"]},
		transition_prob={
			("s1", "a", "s1"): 0.8,
			("s1", "a", "s2"): 0.2,
			("s2", "a", "s2"): 1.0,
			("s3", "a", "s3"): 1.0,
		},
		initial_dist={"s1": 1.0, "s2": 0.0, "s3": 0.0},
		reward_function="s1 + s2 + max(s3, 0)",
	)


def main() -> None:
	mdp = build_small_mdp()
	state_values = {"s1": 0.5, "s2": 0.2, "s3": 0.3}

	reward = mdp.evaluate_reward(state_values)
	print("MDP created and validated successfully")
	print("state_values:", state_values)
	print("reward:", reward)
	print("expected:", 0.5 + 0.2 + max(0.3, 0))


if __name__ == "__main__":
	main()
