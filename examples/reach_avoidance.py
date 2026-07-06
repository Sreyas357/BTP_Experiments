from mdp import MDP


def generate_reach_avoid_mdp(
    rows: int,
    cols: int,
    target_fraction: float = 0.9,
    unsafe_threshold: float = 0.1,
    slip_prob: float = 0.1
):
    """
    Grid reach-avoid:

    - Start: left column center
    - Goal: right column
    - Unsafe: configurable middle strip
    - Dynamics: stochastic (like paper: some probability stays)

    Reward = soft version of reach-avoid:
        + target mass
        - penalty if unsafe mass exceeds threshold
    """

    states = []
    actions = ["right", "up", "down"]

    def state(i, j):
        return f"s_{i}_{j}"

    # --- states ---
    for i in range(cols):
        for j in range(rows):
            states.append(state(i, j))

    # --- define regions ---
    goal_states = [state(cols - 1, j) for j in range(rows)]

    # unsafe: middle columns
    unsafe_states = [
        state(i, j)
        for i in range(cols // 3, 2 * cols // 3)
        for j in range(rows)
    ]

    # --- state-action map ---
    state_action_map = {}
    for i in range(cols):
        for j in range(rows):
            s = state(i, j)

            if i == cols - 1:
                state_action_map[s] = ["right"]  # absorbing
            else:
                acts = ["right"]
                if j > 0:
                    acts.append("down")
                if j < rows - 1:
                    acts.append("up")
                state_action_map[s] = acts

    # --- transitions (stochastic like paper S cells) ---
    transition_prob = {}

    for i in range(cols):
        for j in range(rows):
            s = state(i, j)

            for a in state_action_map[s]:

                if i == cols - 1:
                    transition_prob[(s, a, s)] = 1.0
                    continue

                if a == "right":
                    main = state(i + 1, j)
                elif a == "up":
                    main = state(i, j + 1)
                elif a == "down":
                    main = state(i, j - 1)

                # stochastic: stay with slip_prob
                transition_prob[(s, a, s)] = slip_prob
                transition_prob[(s, a, main)] = 1.0 - slip_prob

    # --- initial distribution ---
    initial_dist = {s: 0.0 for s in states}
    initial_dist[state(0, rows // 2)] = 1.0

    # --- reward (soft reach-avoid) ---
    goal_sum = ", ".join(goal_states)
    unsafe_sum = ", ".join(unsafe_states)

    reward_function = (
        f"{10}*sum([{goal_sum}])"
        f" - {20}*max(0, sum([{unsafe_sum}]) - {unsafe_threshold})"
    )

    return MDP(
        states=states,
        actions=actions,
        state_action_map=state_action_map,
        transition_prob=transition_prob,
        initial_dist=initial_dist,
        reward_function=reward_function,
        is_concave_reward=True
    )