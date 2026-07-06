from pydantic import BaseModel, PrivateAttr, model_validator
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import ast
import copy

def has_duplicates(lst: List):
    return len(lst) != len(set(lst))


ALLOWED_REWARD_IDENTIFIERS = {"max", "min", "abs", "sum"}


class RewardExpressionValidator(ast.NodeVisitor):
    def __init__(self, allowed_variables: Set[str]):
        self.allowed_variables = allowed_variables

    def visit_Name(self, node: ast.Name):
        if node.id in ALLOWED_REWARD_IDENTIFIERS:
            return
        if node.id not in self.allowed_variables:
            raise ValueError(f"Unknown state symbol in reward_function: {node.id}")


class MDP(BaseModel):
    states: List[str]
    actions: List[str]
    state_action_map: Dict[str, List[str]]
    transition_prob: Dict[Tuple[str, str, str], float]
    initial_dist: Dict[str, float]
    reward_function : Optional[str] = None # Ex s1 + s2 + max(s3,0)
    is_max_linear_reward : bool = False
    is_concave_reward : bool = False
    _compiled_reward_expr: Optional[object] = PrivateAttr(default=None)

    reward_model : Any  = None
    float_precision: float = 1e-6

    @model_validator(mode="after")
    def validate_mdp(self):
        states = self.states
        actions = self.actions
        state_action_map = self.state_action_map
        transition_prob = copy.deepcopy(self.transition_prob)  # copy to safely modify
        initial_dist = self.initial_dist

        # --- basic checks ---
        if not states:
            raise ValueError("states must be non-empty")
        if not actions:
            raise ValueError("actions must be non-empty")
        if not initial_dist:
            raise ValueError("initial_dist must be non-empty")

        # --- duplicates ---
        if has_duplicates(states):
            raise ValueError("Duplicate states")
        if has_duplicates(actions):
            raise ValueError("Duplicate actions")

        # --- exact key matching ---
        if set(state_action_map.keys()) != set(states):
            raise ValueError("state_action_map must match states exactly")

        if set(initial_dist.keys()) != set(states):
            raise ValueError("initial_dist must match states exactly")

        # --- initial distribution ---
        for s, p in initial_dist.items():
            if not (0 <= p <= 1):
                raise ValueError(f"Invalid probability {p} at state {s}")

        if abs(sum(initial_dist.values()) - 1) > self.float_precision:
            raise ValueError("Initial distribution must sum to 1")

        # --- state_action_map checks ---
        for s, action_list in state_action_map.items():
            if not action_list:
                raise ValueError(f"No actions defined for state {s}")

            if has_duplicates(action_list):
                raise ValueError(f"Duplicate actions in state {s}")

            for a in action_list:
                if a not in actions:
                    raise ValueError(f"Unknown action {a} in state {s}")

        # --- transition checks ---
        total_prob = {}
        reachable_states = {}

        for (src, action, dst), prob in transition_prob.items():
            if src not in states:
                raise ValueError(f"Unknown state {src}")
            if dst not in states:
                raise ValueError(f"Unknown state {dst}")
            if action not in actions:
                raise ValueError(f"Unknown action {action}")
            if action not in state_action_map[src]:
                raise ValueError(f"Invalid action {action} for state {src}")

            if not (0 <= prob <= 1):
                raise ValueError(f"Invalid probability {prob} for {(src, action, dst)}")

            key = (src, action)
            total_prob[key] = total_prob.get(key, 0) + prob
            reachable_states.setdefault(key, []).append(dst)

        # --- ensure completeness ---
        for src in states:
            for action in state_action_map[src]:
                key = (src, action)

                if key not in total_prob:
                    raise ValueError(f"Missing transitions for {(src, action)}")

                if abs(total_prob[key] - 1) > self.float_precision:
                    raise ValueError(f"Probabilities do not sum to 1 for {(src, action)}")

                for dst in states:
                    if dst not in reachable_states[key]:
                        transition_prob[(src, action, dst)] = 0.0

        # assign back modified transitions
        self.transition_prob = transition_prob

        if self.reward_model is None:
            if self.reward_function is None:
                raise ValueError("Either reward_model or reward_function must be provided")
            self.validate_reward_function()

        return self
    
    def validate_reward_function(self):
        expr = self.reward_function.strip()
        if not expr:
            raise ValueError("reward_function must be non-empty")

        try:
            parsed = ast.parse(expr, mode="eval")
        except SyntaxError as exc:
            raise ValueError(f"Invalid reward_function syntax: {exc}") from exc

        RewardExpressionValidator(set(self.states)).visit(parsed)
        self._compiled_reward_expr = compile(parsed, "<reward_function>", "eval")
        return True

    def evaluate_reward(self, state_values: Dict[str, float]) -> float:
        if self._compiled_reward_expr is None:
            self.validate_reward_function()

        missing_states = set(self.states) - set(state_values.keys())
        if missing_states:
            raise ValueError(
                f"Missing values for states in reward evaluation: {sorted(missing_states)}"
            )

        invalid_prob_states = [
            s for s in self.states if not (0.0 <= float(state_values[s]) <= 1.0)
        ]
        if invalid_prob_states:
            raise ValueError(
                f"State values must be probabilities in [0, 1]: {sorted(invalid_prob_states)}"
            )

        total_mass = sum(float(state_values[s]) for s in self.states)
        if abs(total_mass - 1.0) > self.float_precision:
            raise ValueError("State values must sum to 1")

        local_scope = {state: float(state_values[state]) for state in self.states}
        local_scope.update({"max": max, "min": min, "abs": abs, "sum": sum})

        return float(eval(self._compiled_reward_expr, {"__builtins__": {}}, local_scope))
