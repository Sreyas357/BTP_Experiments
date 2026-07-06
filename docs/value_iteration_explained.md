# Value Iteration Solver (Max-Linear Distributional Rewards) — Code Walkthrough

This file explains how the solver in `solvers/value_iteration.py` works, with the key ideas aligned to the report **“Distributional Rewards in MDPs” (March 26, 2026)**.

The goal is to make the `solve()` implementation and the surrounding helper methods easy to understand:

- what the **Γ-sets / alpha-vectors** are,
- what **“dominated” / Pareto pruning** means here,
- how the Bellman backup becomes a max over linear functions,
- and how the code keeps the computation tractable.

---

## 1) What problem this solver is solving

### Distributional state

Instead of optimizing over a single (joint) state, we optimize over a **distribution** over states:

- $\mu \in \Delta(S)$, the probability simplex over the finite state set $S$.
- $\mu[s]$ is the fraction/probability mass in state $s$.

Given a deterministic policy $\pi$ that chooses one action per state, the distribution evolves **linearly**:

$$
\mu' = T_\pi^\top \mu
$$

where $T_\pi$ is the $|S|\times|S|$ transition matrix induced by policy $\pi$.

### Max-linear (piecewise-linear convex) reward

The report’s Section 7 focuses on convex rewards that are **max-linear** in the distribution:

$$
R(\mu) = \max_i a_i^\top \mu \quad (\text{possibly plus constants})
$$

This code implements that idea via `mdp.reward_model`, which is a list of terms:

- `LinearReward`: contributes one linear piece $a^\top\mu + b$.
- `MaxLinearReward`: contributes a _max over multiple linear pieces_, scaled by a nonnegative weight.

When you combine multiple terms, the total reward is still representable as a max over finitely many linear “modes” (more on that in Section 4).

### Objective (finite-horizon, discounted)

For a horizon `horizon = T` and discount `gamma = \gamma`, the solver computes a finite-horizon optimal value function

$$
V_t^*(\mu),\quad t \in \{0,\dots,T\}
$$

with terminal condition $V_T^*(\mu)=0$.

The report gives the Bellman recursion (Proposition 6):

$$
V_t^*(\mu) = \max_{\pi \in \Pi} \Big(R(\mu) + \gamma\, V_{t+1}^*(T_\pi^\top\mu)\Big).
$$

Important detail about _this implementation_:

- In `solvers/value_iteration.py`, the immediate reward is treated as a function of $\mu$ only, **not of the chosen policy**. That’s why `_best_policy_idx()` can ignore the current reward when selecting a policy (Section 7 below).

---

## 2) Key representation: value functions as maxima of linear forms (Γ-sets)

A central theorem in the report (Theorem 7) is:

$$
V_t^*(\mu) = \max_{v \in \Gamma_t} v^\top\mu
$$

for some finite set $\Gamma_t \subset \mathbb{R}^{|S|}$.

In the code:

- `gamma_sets[t]` is exactly this $\Gamma_t$.
- It is stored as a 2D NumPy array of shape `[K_t, S]`.
- Each row is one “alpha-vector” (the report’s $v$).

So computing the value at a belief/distribution is just:

- convert `mu: Dict[str,float]` → vector `mu_vec` of length `S`
- return `max_k gamma_sets[t][k] @ mu_vec`.

That’s what `value()` does.

---

## 3) “Folded” alpha vectors (why the code stores `v̄ = v + c·1`)

The solver stores **folded alpha-vectors**:

$$
\bar v = v + c\,\mathbf{1}
$$

where $\mathbf{1}$ is the all-ones vector.

Why this is valid:

For any distribution $\mu \in \Delta(S)$ we have $\sum_s \mu[s] = 1$, so

$$
\bar v^\top \mu = (v + c\mathbf{1})^\top\mu = v^\top\mu + c\,\underbrace{\mathbf{1}^\top\mu}_{=1} = v^\top\mu + c.
$$

So a constant offset can be “absorbed” into the vector without changing evaluation on the simplex.

In the code:

- reward modes are built as `(a_mode, b_mode)` pairs,
- then “folded” to a vector `ā = a + b·1`.

See `_build_reward_modes_folded()`.

A second reason folding is convenient: it stays closed under the Bellman backup because policy transition matrices are **row-stochastic** (they map distributions to distributions), which implies $T_\pi\mathbf{1}=\mathbf{1}$.

---

## 4) Reward “modes” (how `reward_model` becomes a finite list of linear vectors)

The report’s Algorithm 1 assumes a finite set of reward vectors $a_{\pi,i}$ and then loops over:

- policy $\pi$,
- reward mode $i$,
- next-step alpha-vector $v \in \Gamma_{t+1}$.

This code constructs the “reward mode” list in `_build_reward_modes()`:

- Start with one base mode: `(0-vector, 0)`.
- For each `LinearReward`, add its weights to every existing mode.
- For each `MaxLinearReward` with pieces $(a_j, b_j)$, **expand** the mode list by taking the cartesian product with those pieces.

So the final reward is represented as a max over modes $m \in \{1,\dots,M\}$:

$$
R(\mu) = \max_m \big(a_m^\top\mu + b_m\big)
$$

and the code uses the folded form:

- `self._R_bar` has shape `[M, S]` and stores $\bar a_m = a_m + b_m\mathbf{1}$.

Note: the report allows reward vectors to depend on policy ($a_{\pi,i}$). This implementation currently uses $a_i$ independent of $\pi$ (equivalent to $a_{\pi,i}=a_i$ for all $\pi$).

---

## 5) Enumerating deterministic policies and building transition matrices

### Enumerating policies

A deterministic policy here is a mapping “state → action”:

$$
\pi: S \to A.
$$

The code enumerates **all** deterministic policies:

- `self.policies = self._enumerate_deterministic_policies()`
- it does `product(*action_lists)` across states

So the number of deterministic policies is

$$
|\Pi| = \prod_{s\in S} |A(s)|.
$$

This can explode quickly, so this solver is intended for small problems (or problems with few actions per state).

### Transition matrices for policies

For each policy $\pi$, the induced transition matrix is

$$
[T_\pi]_{s,s'} = P(s' \mid s, \pi(s)).
$$

The code precomputes these into a single stacked tensor:

- `T_stack[p, s, s']` has shape `[P, S, S]`.

This makes the Bellman backup vectorized.

---

## 6) The Bellman backup in max-linear form (how `solve()` matches Algorithm 1)

Assume inductively that

$$
V_{t+1}(\mu') = \max_{k} \bar v_k^\top \mu'
$$

and the reward is

$$
R(\mu) = \max_m \bar a_m^\top \mu.
$$

Then:

$$
\begin{align}
V_t(\mu)
&= R(\mu) + \gamma \max_\pi V_{t+1}(T_\pi^\top\mu)\\
&= \max_{m} \bar a_m^\top\mu + \gamma\max_\pi \max_{k} \bar v_k^\top (T_\pi^\top\mu)\\
&= \max_{m,\pi,k} \Big(\bar a_m + \gamma T_\pi\bar v_k\Big)^\top\mu.
\end{align}
$$

So the new Γ-set is built by taking all candidates:

$$
\Gamma_t \leftarrow \{\bar a_m + \gamma T_\pi \bar v_k\ \mid\ m,\pi,k\}.
$$

This is exactly what `solve()` does:

1. Start from terminal set: `gamma_sets[T] = {0}`.
2. For `t = T-1 .. 0`:
   - take `G_next = gamma_sets[t+1]` (shape `[K,S]`)
   - form all vectors `R_bar[m] + gamma * T_pi @ G_next[k]`.

### How the code vectorizes the candidate construction

Inside `solve()`:

- `batch_future = gamma * (T_batch @ G_next.T).transpose(0,2,1)`

Shapes:

- `T_batch`: `[b, S, S]`
- `G_next.T`: `[S, K]`
- `T_batch @ G_next.T`: `[b, S, K]`
- transpose → `[b, K, S]`

Then it adds reward modes by broadcasting:

- `R_bar[:, None, None, :] + batch_future[None, ...]`
- yields `[M, b, K, S]`
- reshaped into `[M*b*K, S]` candidate vectors.

---

## 7) What “Pareto front” / “dominated vectors” means here

After generating candidates for $\Gamma_t$, the report suggests an optional step:

> Remove dominated vectors from $\Gamma_t$ (Remark 5).

A vector $w$ is dominated if there exists $w'$ such that

$$
(w')^\top\mu \ge w^\top\mu \quad \forall \mu\in\Delta(S).
$$

### Key simplification for distributions over states

Because $\mu$ ranges over the simplex, the worst-case (minimum) of a linear function over $\Delta(S)$ happens at a vertex (a one-hot distribution). This implies:

$$
(w' - w)^\top\mu \ge 0\ \forall\mu\in\Delta(S)
\quad\Longleftrightarrow\quad
w'[s] \ge w[s]\ \forall s.
$$

So “dominance over all distributions” reduces to **componentwise dominance**.

This is why the code can prune dominated vectors using simple array comparisons—no linear programs.

### Why this is called a Pareto front

In multi-objective optimization, a point is Pareto-dominated if another point is at least as good in every coordinate. Here:

- each vector is a point in $\mathbb{R}^{|S|}$,
- “better” means “greater in every state coordinate”,
- and the undominated set is the **Pareto front** under that partial order.

In the code:

- the maintained `pareto_front` is a set of vectors that are not componentwise dominated by any other kept vector (up to tolerance).

### Important nuance: this pruning is safe but not always minimal

Componentwise dominance is a sufficient condition for “never useful”.

There can exist vectors that are **not** componentwise dominated but still never achieve the max $\max_v v^\top\mu$ for any $\mu$ (they lie below the upper envelope). Removing those requires more advanced “witness-region” / LP-based pruning (common in POMDP literature). This solver does **not** do that; it uses the cheap, always-correct dominance rule.

---

## 8) How `solve()` maintains the Pareto front incrementally (batching)

Naively, for each $t$ you would build the full candidate set of size

$$
|\Pi| \cdot M \cdot |\Gamma_{t+1}|
$$

and then prune it. If this intermediate set is huge, even the pruning step becomes expensive.

This code avoids building a massive intermediate list by processing policies in **batches** (`policy_batch_size`).

### Incremental merge strategy

For each batch of policies, it:

1. generates candidates for those policies only
2. merges them into the current Pareto front using `_merge_into_pareto()`

This keeps the intermediate set closer to “(batch candidates) + (current front)” rather than “(all candidates)”.

### `_merge_into_pareto(new_cands, front)`

Given:

- `front` is already Pareto-pruned,
- `new_cands` are unpruned candidates from a batch,

it does:

1. **Drop new candidates dominated by the current front**
2. **Drop old front vectors dominated by surviving new ones**
3. **Prune within the surviving new candidates**
4. concatenate.

This is usually much cheaper than pruning the entire combined set from scratch.

### Numerical tolerance

Dominance comparisons use `prune_tol` to treat near-equality as dominance, preventing numerical noise from creating duplicate/near-duplicate vectors.

Also `_prune_dominated()` rounds candidates to 12 decimals before `np.unique()` to deduplicate cheaply.

---

## 9) Optional PBVI-style approximate pruning (`max_vectors`)

If `max_vectors` is set, after exact dominance pruning the solver further reduces the Γ-set with a **point-based** heuristic:

- sample belief points $B \subset \Delta(S)$ (Dirichlet by default)
- for each belief point, keep the alpha-vector that maximizes $b^\top v$

Implementation: `_pbvi_prune()`.

Interpretation:

- This keeps vectors that are “best somewhere” on the sampled belief set.
- It does **not** guarantee exact optimality on all $\mu$, but controls the growth of `|Γ_t|`.

The code also has a tie-break if too many winners exist:

- keep the `max_vectors` vectors with the highest “coverage” (how many belief points chose them).

---

## 10) Reading the public API methods

### `solve() → gamma_sets`

Returns a list of length `horizon+1`.

- `gamma_sets[t]` is a 2D array `[K_t, S]` of folded alpha vectors.
- terminal set is `[ [0,0,...,0] ]`.

### `value(mu, t=0)`

Computes:

$$
V_t(\mu) = \max_k \Gamma_t[k]^\top\mu.
$$

### `best_policy_for_distribution(mu, t)`

Returns one of the enumerated deterministic policies `π(s)`.

Internally it calls `_best_policy_idx(mu_vec, t)`.

### `_best_policy_idx(mu_vec, t)`

This method chooses the policy by maximizing the **next-step value**:

$$
\arg\max_\pi V_{t+1}(T_\pi^\top\mu).
$$

Why it can ignore current reward:

- in this solver, $R(\mu)$ does not depend on $\pi$.

Implementation details:

- it computes all pushed distributions `pushed_mus[p] = T_p^T mu` at once using `einsum`
- then evaluates `max_k G_next[k] @ pushed_mus[p]` for each policy.

### `rollout_distribution(mu0)`

Greedily applies `best_policy_for_distribution` at each time step and propagates the distribution:

$$
\mu_{t+1} = T_{\pi_t(\mu_t)}^\top \mu_t.
$$

This gives a trajectory of distributions $\mu_0,\dots,\mu_T$.

---

## 11) Practical notes / limitations

- **Policy explosion:** enumerating all deterministic policies is exponential in `|S|`.
- **Reward assumptions:** this solver assumes the reward terms can be represented as max-linear in the _state distribution_ $\mu$ via `reward_model`.
- **Pruning is dominance-only:** safe and fast, but may keep extra vectors that are never optimal.
- **PBVI is approximate:** use it when Γ-sets get too large.

---

## 12) Quick mental model (how to think about Γ-sets)

At each time step $t$:

- `gamma_sets[t]` stores many candidate “linear value functions”.
- For a specific $\mu$, you evaluate all of them and take the maximum.
- The Pareto/dominance pruning removes vectors that are uniformly worse for every possible distribution.

This matches the report’s Algorithm 1: build candidates by combining reward modes, policies, and next-step vectors, then prune.

---

## 13) Worked toy example (2 states) — Γ-sets, Pareto, dominance

Let $S=\{s_1,s_2\}$.

Any distribution $\mu\in\Delta(S)$ can be written as

$$
\mu = (p,1-p),\quad p\in[0,1].
$$

### 13.1 Evaluating a Γ-set

Suppose at some time $t$ we have

$$
\Gamma_t = \{v^{(1)}, v^{(2)}\},\quad
v^{(1)}=(1,0),\ v^{(2)}=(0,1).
$$

Then

$$
V_t(\mu)=\max\{(1,0)\cdot(p,1-p),\ (0,1)\cdot(p,1-p)\}
=\max\{p, 1-p\}.
$$

Interpretation:

- if $p\ge 1/2$, the best vector is $v^{(1)}$;
- if $p\le 1/2$, the best vector is $v^{(2)}$.

This is exactly what the code does with `gamma_sets[t] @ mu_vec` followed by `max`.

### 13.2 Componentwise dominance (the pruning rule used in code)

Consider two candidate vectors

$$
w=(0.8,0.8),\qquad w'=(1.0,1.0).
$$

We have $w'[s]\ge w[s]$ for both states, i.e. $w'\ge w$ componentwise.
Therefore, for every distribution $\mu=(p,1-p)$:

$$
(w')^\top\mu \ge w^\top\mu.
$$

So $w$ is **dominated** and can be removed with no effect on $V_t(\cdot)$.

This is the exact check implemented by `_prune_dominated()` and `_merge_into_pareto()`:

- “$w'$ dominates $w$” ⇔ $w' - w \ge 0$ coordinatewise (up to tolerance).

### 13.3 Why “Pareto front” terminology fits

If you view a vector $v=(v[s_1], v[s_2])$ as a point in the plane, then

- $v'$ dominates $v$ if it is up-and-right of it (at least as large in both coordinates).
- the set of non-dominated points is the **Pareto front**.

The solver maintains this front incrementally while building $\Gamma_t$.

### 13.4 Important nuance: not dominated, but still never optimal

Dominance pruning is safe, but it is not the strongest possible pruning.

Example: keep the earlier

$$
v^{(1)}=(1,0),\ v^{(2)}=(0,1)
$$

and add

$$
v^{(3)}=(0.4,0.4).
$$

Is $v^{(3)}$ dominated by $v^{(1)}$? No, because $v^{(1)}[s_2]=0 < 0.4$.
Dominated by $v^{(2)}$? No, because $v^{(2)}[s_1]=0 < 0.4$.

So dominance pruning will keep $v^{(3)}$.

But it is **never optimal**:

$$
v^{(3)}\cdot(p,1-p)=0.4
$$

while

$$
\max\{v^{(1)}\cdot\mu, v^{(2)}\cdot\mu\}=\max\{p,1-p\}\ge 0.5
$$

for all $p\in[0,1]$. So $v^{(3)}$ never attains the maximum and could be removed by a stronger (but more expensive) pruning method.

This is why the doc earlier says: “dominance-only pruning may keep extra vectors that are never optimal.”
