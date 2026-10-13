# Aircraft recovery: formulation

Reference formulation for this repository; the README condenses it. The last column of each data table gives the name used in code.

Notation: $`[P]`$ is 1 if the statement $`P`$ is true and 0 otherwise.

## 1. Sets and data

**Sets**

| Symbol | Meaning | Code |
|---|---|---|
| $`K`$ | aircraft (single fleet type) | `K` |
| $`S`$ | stations (airports) | `S` |
| $`F`$ | flights still to operate at the recovery time (see §2) | `F` |
| $`D=\lbrace 0,\Delta,2\Delta,\dots,D_{\max}\rbrace`$ | allowed delays, step $`\Delta`$ (default 15 min) | `D` |

**Flight data**, for $`f\in F`$

| Symbol | Meaning | Code |
|---|---|---|
| $`o_f,\ \delta_f`$ | origin and destination, $`o_f\neq\delta_f`$ | `origin_f`, `dest_f` |
| $`t_f`$ | scheduled departure (off-block), minutes from 00:00 | `dep_f` |
| $`\beta_f>0`$ | block time, off-block to on-block | `blk_f` |
| $`p_f`$ | passengers | `pax_f` |
| $`k^0_f`$ | originally assigned aircraft | `orig_f` |

**Operations and disruptions**

| Symbol | Meaning | Code |
|---|---|---|
| $`\tau>0`$ | minimum turnaround, on-block to next off-block | `TAT` |
| $`b_s`$ | aircraft whose planned rotation ends at $`s`$, with $`\sum_s b_s=\lvert K\rvert`$ | `b_s` |
| $`\hat k`$, $`[t_1,t_2)`$ | AOG aircraft and its unavailability window; it can leave again at $`t_2`$ | |
| $`\bar s`$, $`[t_a,t_b)`$ | closed station and closure window | |
| $`t_0`$ | recovery time: $`t_1`$ for an AOG, $`t_a`$ for a closure | `t0` |

**Costs**

| Symbol | Meaning | Code |
|---|---|---|
| $`c^{\text{del}}\ge 0`$ | cost per passenger-minute of delay | `c_delay` |
| $`C_f=c^{\text{canc}}_0+c^{\text{canc}}_1 p_f>0`$ | cost of cancelling $`f`$ | `c_cancel_fixed`, `c_cancel_pax` |
| $`c^{\text{swap}}\ge 0`$ | cost per flight flown by an aircraft other than $`k^0_f`$ | `c_swap` |
| $`c^{\text{pos}}>0`$ | cost per aircraft missing at a station at end of day | `c_pos` |

## 2. Preprocessing

### 2.1 Recovery time and frozen flights

Recovery starts at $`t_0`$. Flights with $`t_f<t_0`$ have already departed, so they are fixed data, not decisions:

$$
F=\lbrace f : t_f\ge t_0\rbrace .
$$

Frozen flights are assumed to have operated on schedule.

In words: the solver only re-plans what has not happened yet. Frozen flights matter only through where each aircraft is and when it is free.

### 2.2 Ready state of each aircraft

Let $`\bar f_k`$ be the last frozen flight of aircraft $`k`$.

$$
(s_k,\ \rho_k)=
\begin{cases}
\big(\text{start-of-day station of }k,\ \ t_0\big) & \text{if }k\text{ has no frozen flight}\\
\big(\delta_{\bar f_k},\ \ \max(t_0,\ t_{\bar f_k}+\beta_{\bar f_k}+\tau)\big) & \text{otherwise}
\end{cases}
\qquad\qquad
\rho_{\hat k}\leftarrow\max(\rho_{\hat k},\ t_2)
$$

In words: $`s_k`$ is where aircraft $`k`$ is when recovery starts, and $`\rho_k`$ is the earliest time it can take off again. That is either now, or after its last flight has landed and turned around. The AOG aircraft cannot leave before it is fixed at $`t_2`$.

### 2.3 Delay grid and the elementarity cap

$$
\bar D=\min_{f,g\in F:\ \delta_f=o_g}\big(\beta_f+\beta_g\big)+2\tau,
\qquad
D_{\max}<\bar D .
$$

If no pair $`(f,g)`$ with $`\delta_f=o_g`$ exists, set $`\bar D=+\infty`$.

**Claim.** If $`D_{\max}<\bar D`$, no route (§4) contains two copies of the same flight.

**Proof.** Suppose a path in the network of §3 visits $`(f,d)`$ and later $`(f,d')`$. After $`f`$ it must fly some $`g`$ with $`o_g=\delta_f`$. Departure times increase along the path by at least the block time plus $`\tau`$ at each step, so

$$
t_f+d' \ \ge\ t_f+d+\beta_f+\tau+\beta_g+\tau
\quad\Longrightarrow\quad
d'-d\ \ge\ \beta_f+\beta_g+2\tau\ \ge\ \bar D\ >\ D_{\max}.
$$

This is impossible, because $`d,d'\in[0,D_{\max}]`$. $`\blacksquare`$

In words: to fly $`f`$ twice, the aircraft has to fly $`f`$, come back, and fly $`f`$ again. That takes at least two block times and two turnarounds. If the largest delay is shorter than that, the second copy would need a delay outside the grid.

## 3. Connection network of aircraft $k$

$`G_k=(N_k,A_k)`$ is a graph whose paths are exactly the days aircraft $`k`$ can physically fly.

**Nodes**

- a source $`\text{src}_k`$: aircraft $`k`$ at $`s_k`$, ready at $`\rho_k`$;
- a flight copy $`(f,d)`$ for every $`f\in F`$ and $`d\in D`$, meaning "$`f`$ departs $`d`$ minutes late". It departs at $`t_f+d`$ and arrives at $`t_f+d+\beta_f`$;
- a sink $`\text{snk}_s`$ for every $`s\in S`$, meaning "$`k`$ ends the day at $`s`$".

**Closure.** Under an airport closure, delete the copies that use $`\bar s`$ during the window:

$$
X=\lbrace (f,d) : o_f=\bar s,\ t_f+d\in[t_a,t_b)\rbrace\ \cup\ \lbrace (f,d) : \delta_f=\bar s,\ t_f+d+\beta_f\in[t_a,t_b)\rbrace .
$$

**Arcs**

| Arc | Exists if | Meaning |
|---|---|---|
| $`\text{src}_k\to(f,d)`$ | $`o_f=s_k`$ and $`t_f+d\ge\rho_k`$ | first flight of the day |
| $`(f,d)\to(g,e)`$ | $`\delta_f=o_g`$ and $`t_f+d+\beta_f+\tau\le t_g+e`$ | land, turn around, fly $`g`$ |
| $`(f,d)\to\text{snk}_{\delta_f}`$ | always | $`f`$ is the last flight |
| $`\text{src}_k\to\text{snk}_{s_k}`$ | always | stay on the ground (empty route) |

Arcs touching a deleted copy in $`X`$ are removed with it.

**$`G_k`$ is acyclic.** A connection arc requires $`t_g+e\ge t_f+d+\beta_f+\tau>t_f+d`$, so departure times strictly increase along every path. Ordering the nodes as source, then copies by departure time, then sinks gives a topological order.

Copy-to-copy arcs do not depend on $`k`$. Only the start arcs, the empty route and the swap costs do.

In words: you can only chain flights that start where the previous one landed, after the turnaround. Time only moves forward, so the graph has no cycles.

## 4. Routes

A route $`r\in R_k`$ is a path from $`\text{src}_k`$ to some sink in $`G_k`$. For each route:

$$
a_{fr}=[ r\text{ contains a copy of }f ],
\qquad
e_{sr}=[ r\text{ ends at }\text{snk}_s ],
$$

$$
c_r=\sum_{(f,d)\in r}\Big(c^{\text{del}} p_f d\ +\ c^{\text{swap}} [ k^0_f\neq k ]\Big).
$$

By §2.3, $`a_{fr}\in\lbrace 0,1\rbrace`$ also equals the number of copies of $`f`$ in $`r`$. The empty route has $`a_{fr}=0`$ for all $`f`$, ends at $`s_k`$, and costs 0.

In words: a route is one complete day plan for one aircraft: which flights, with which delays, and where it sleeps. Its cost is the delay cost of its flights plus a swap penalty for every flight it takes over from another aircraft.

## 5. Master problem

**Variables**

| Variable | Meaning |
|---|---|
| $`\lambda_r\in\lbrace 0,1\rbrace`$ | aircraft $`k`$ flies route $`r\in R_k`$ |
| $`z_f\in\lbrace 0,1\rbrace`$ | flight $`f`$ is cancelled |
| $`y_s\ge 0`$ | aircraft missing at $`s`$ at end of day |

**Model**

$$
\begin{aligned}
\min\quad & \sum_{k\in K}\sum_{r\in R_k}c_r\lambda_r\ +\ \sum_{f\in F}C_f z_f\ +\ \sum_{s\in S}c^{\text{pos}}y_s\\
\text{s.t.}\quad
& \sum_{k\in K}\sum_{r\in R_k}a_{fr}\lambda_r+z_f=1 && \forall f\in F && \text{(cover)}\\
& \sum_{r\in R_k}\lambda_r=1 && \forall k\in K && \text{(one route per aircraft)}\\
& \sum_{k\in K}\sum_{r\in R_k}e_{sr}\lambda_r+y_s\ge b_s && \forall s\in S && \text{(positioning)}\\
& \lambda_r\in\lbrace 0,1\rbrace,\quad z_f\in\lbrace 0,1\rbrace,\quad y_s\ge 0
\end{aligned}
$$

In words:

- **Cover:** every flight is flown by exactly one aircraft, or cancelled.
- **One route per aircraft:** every aircraft gets exactly one day plan, and staying on the ground counts as a plan.
- **Positioning:** if fewer aircraft than planned end at $`s`$, pay $`c^{\text{pos}}`$ for each missing one. Only shortfalls need a penalty: the fleet size is fixed, so a surplus somewhere means a shortfall elsewhere. At an optimum, $`y_s=\max\big(0,\ b_s-\sum_{k,r}e_{sr}\lambda_r\big)`$.

**Why the model is correct.** Each aircraft's own feasibility (stations, turnarounds, ready time, closures) is built into its network, so every route is flyable. The only interaction between aircraft is that no flight is flown twice, and the cover rows enforce exactly that. So feasible recovery plans correspond one to one to feasible $`(\lambda,z,y)`$ with $`y_s=\max\big(0,\ b_s-\sum_{k,r}e_{sr}\lambda_r\big)`$, with the same cost. A larger $`y_s`$ is also feasible but only adds cost, so every optimum uses this value.

**LP relaxation.** Replace the binaries by $`\lambda,z\ge 0`$; the upper bounds of 1 follow from the cover and one-route rows. Its optimal value is $`z_{\text{LP}}`$.

**Restricted master (RMP).** The same LP over a subset $`\bar R_k\subseteq R_k`$ that always contains the empty route, with all $`z_f`$ and $`y_s`$ present. Its optimal value is $`z_{\text{RMP}}\ge z_{\text{LP}}`$, because it is a restriction.

## 6. LP dual of the master

### 6.1 How the dual is built

Attach a multiplier to every row: $`\pi_f`$ to the cover rows, $`\sigma_k`$ to the one-route rows, and $`\mu_s`$ to the positioning rows. Multiply each row by its multiplier and add all the rows up.

- **Equality rows** can be multiplied by any number, so $`\pi_f`$ and $`\sigma_k`$ are free.
- **Positioning rows** are $`\ge`$, so they keep their direction only if $`\mu_s\ge 0`$.

So for $`\pi,\sigma`$ free and $`\mu\ge 0`$, every LP-feasible $`(\lambda,z,y)`$ satisfies

$$
\sum_{k}\sum_{r\in R_k}\lambda_r\Big(\sum_f a_{fr}\pi_f+\sigma_k+\sum_s e_{sr}\mu_s\Big)+\sum_f z_f \pi_f+\sum_s y_s \mu_s\ \ \ge\ \ \sum_f\pi_f+\sum_k\sigma_k+\sum_s b_s\mu_s .
$$

Now suppose each variable's coefficient on the left is at most its cost. All variables are nonnegative, so then objective $`\ge`$ left side $`\ge`$ right side. The right side is therefore a lower bound on the cost of every feasible point. The dual picks the multipliers that make this bound as large as possible.

### 6.2 The dual

$$
\begin{aligned}
\max\quad & \sum_{f\in F}\pi_f+\sum_{k\in K}\sigma_k+\sum_{s\in S}b_s\mu_s\\
\text{s.t.}\quad
& \sum_f a_{fr}\pi_f+\sigma_k+\sum_s e_{sr}\mu_s\le c_r && \forall k\in K,\ r\in R_k && (\lambda_r)\\
& \pi_f\le C_f && \forall f\in F && (z_f)\\
& \mu_s\le c^{\text{pos}} && \forall s\in S && (y_s)\\
& \pi,\ \sigma\ \text{free},\quad \mu\ge 0
\end{aligned}
$$

**Weak duality:** every dual-feasible point has value $`\le z_{\text{LP}}`$. **Strong duality:** at the optimum, the two values are equal.

### 6.3 Reading the dual constraints off the columns

Each primal column gives one dual constraint. Go down the column, multiply every nonzero coefficient by its row's multiplier, add them up, and require the sum to be at most the column's cost.

| Column | Nonzeros | Dual constraint |
|---|---|---|
| $`\lambda_r`$, $`r\in R_k`$ | 1 in the cover row of each flight of $`r`$; 1 in the one-route row of $`k`$; 1 in the positioning row of $`r`$'s end station | $`\sum_{f\in r}\pi_f+\sigma_k+\mu_{\text{end}(r)}\le c_r`$ |
| $`z_f`$ | 1 in the cover row of $`f`$ | $`\pi_f\le C_f`$ |
| $`y_s`$ | 1 in the positioning row of $`s`$ | $`\mu_s\le c^{\text{pos}}`$ |

| Primal | Dual |
|---|---|
| equality row | free multiplier |
| $`\ge`$ row (minimization) | multiplier $`\ge 0`$ |
| variable $`\ge 0`$ | dual constraint "$`\le`$ its cost" |
| right-hand side of a row | coefficient of that row's multiplier in the dual objective |

The dual has only $`\lvert F\rvert+\lvert K\rvert+\lvert S\rvert`$ variables, but one constraint per route, which is astronomically many. Column generation works in that asymmetry.

### 6.4 The duals as prices

- $`\pi_f`$ is the value of getting flight $`f`$ flown. It never exceeds $`C_f`$, because cancelling is always possible.
- $`\mu_s\ge 0`$ is the value of one more aircraft ending at $`s`$. It is nonnegative because raising $`b_s`$ can only raise the cost, and it never exceeds $`c^{\text{pos}}`$, because a shortfall can always be paid for instead.
- $`-\sigma_k\ge 0`$ is the marginal value of aircraft $`k`$. The empty route's dual constraint reads $`\sigma_k+\mu_{s_k}\le 0`$, so $`\sigma_k\le-\mu_{s_k}\le 0`$ always. At an optimum of the RMP, $`-\sigma_k`$ is the most that any route of $`k`$ in the RMP collects beyond its cost, $`\sum_{f\in r}\pi_f+\mu_{\text{end}(r)}-c_r`$. It is only a marginal value: in `examples/cg_by_hand.py`, removing spare P raises the LP optimum from 30 to 110, while valid final duals give $`\sigma_P=0`$ or $`\sigma_P=-5`$.

A route's dual constraint says that the value it collects (prices of its flights and of its end station) cannot exceed what it costs to fly plus $`-\sigma_k`$, the value its aircraft could produce elsewhere.

**Degeneracy.** Set-partitioning masters are highly degenerate, so the optimal duals are often not unique. Different solvers, or different iterations, may return different prices for the same $`z_{\text{RMP}}`$. This changes which columns get generated, but not $`z_{\text{LP}}`$. It is the main cause of slow convergence ("tailing-off"), and the reason dual stabilization techniques exist.

## 7. Reduced cost, optimality and pricing

### 7.1 Reduced cost and the optimality test

The reduced cost of route $`r\in R_k`$ is the slack of its dual constraint:

$$
\bar c_r=c_r-\sum_f a_{fr}\pi_f-\sigma_k-\sum_s e_{sr}\mu_s .
$$

**Claim.** Let $`(\pi,\sigma,\mu)`$ be optimal duals of the RMP. If $`\bar c_r\ge 0`$ for every $`r\in R_k`$ and every $`k`$, including routes not in the RMP, then $`z_{\text{RMP}}=z_{\text{LP}}`$.

**Proof.** Every RMP contains all $`z_f`$ and $`y_s`$, so its duals satisfy $`\pi_f\le C_f`$, $`0\le\mu_s\le c^{\text{pos}}`$. Together with $`\bar c_r\ge 0`$ for all routes, $`(\pi,\sigma,\mu)`$ is feasible for the full dual of §6.2. Its value is $`z_{\text{RMP}}`$ by strong duality on the RMP, so weak duality gives $`z_{\text{RMP}}\le z_{\text{LP}}`$. The RMP is a restriction, so $`z_{\text{LP}}\le z_{\text{RMP}}`$. $`\blacksquare`$

In words: if no route anywhere is worth more than it costs at the current prices, the prices certify that nothing outside the RMP can improve it.

### 7.2 Pricing is a shortest path

Pricing finds $`\bar c_k^{\ast}=\min_{r\in R_k}\bar c_r`$ for each aircraft. Put weights on $`G_k`$:

$$
w_k(f,d)=c^{\text{del}} p_f d+c^{\text{swap}} [ k^0_f\neq k ]-\pi_f\ \ \text{on node }(f,d),
\qquad
-\mu_s\ \ \text{on arcs into }\text{snk}_s .
$$

Each flight appears at most once on a route (§2.3), so for a route $`r`$ ending at $`s`$ the reduced cost is exactly

$$
\bar c_r=\sum_{(f,d)\in r}w_k(f,d)\ -\ \mu_s\ -\ \sigma_k .
$$

**Dynamic program.** Process the copies in increasing departure time (a topological order, §3). Let $`L_k(v)`$ be the cheapest weight of a path from $`\text{src}_k`$ to copy $`v`$, including $`v`$:

$$
L_k(v)=w_k(v)+\min\Big(\lbrace 0 : \text{src}_k\to v\in A_k\rbrace\ \cup\ \lbrace L_k(u) : u\to v\in A_k\rbrace\Big),
\qquad \min\emptyset=+\infty .
$$

Then

$$
\bar c_k^{\ast}=\min\Big(-\mu_{s_k},\ \ \min_{v=(f,d)}\big(L_k(v)-\mu_{\delta_f}\big)\Big)-\sigma_k ,
$$

where the first term is the empty route. The minimizing path is recovered by storing, for each $`v`$, the predecessor that achieved the minimum. The running time is $`O(\lvert N_k\rvert+\lvert A_k\rvert)`$.

**Implementation.** The in-arcs of a copy $`v=(g,e)`$ come from the copies that land at $`o_g`$ and are turned around by $`t_g+e`$, plus the start arc if $`o_g=s_k`$ and $`t_g+e\ge\rho_k`$. So the inner minimum of the recurrence is a running minimum per station. Treat $`\text{src}_k`$ as a copy with $`L_k=0`$ that is ready at $`s_k`$ at time $`\rho_k`$. Sweep two kinds of events in time order: copy $`(f,d)`$ becomes ready at $`\delta_f`$ at time $`t_f+d+\beta_f+\tau`$, and copy $`(f,d)`$ departs at $`t_f+d`$; on ties, ready events go first, because a connection may leave exactly when the turnaround ends. A ready event lowers the running minimum of its station; a departure reads it. The sweep computes the same $`L_k`$ as the recurrence without listing the arcs, in $`O(\lvert N_k\rvert)`$ per call after one sort of the events per network, instead of $`O(\lvert A_k\rvert)`$.

**Why the dynamic program is correct.** A path's weight is a sum over its nodes. Which nodes can follow $`v`$ depends only on $`v`$ (a place and a time), not on how the aircraft got there. So the best path through $`v`$ always starts with a best path to $`v`$ (Bellman's principle). At each node only the cheapest prefix is kept, and all others are discarded for good.

In words: given the prices, each aircraft looks for the day that collects the most flight value minus cost. The number of routes grows exponentially with the length of the day, but the dynamic program touches each arc once. This is also why repeated flights had to be ruled out: if the future depended on which flights were already flown, a node would no longer be enough state.

## 8. Column generation

0. **Initialize.** For each $`k`$, add the empty route and the original rotation made feasible. Walk $`k`$'s remaining planned flights in order and give each the smallest delay $`d\in D`$ that respects the ready time ($`\rho_k`$ for the first flight, the previous landing plus $`\tau`$ after that) and whose copy is not deleted by a closure. Cut the rotation at the first flight with no such delay. Add all $`z_f`$ and $`y_s`$. This RMP is always feasible: cancel everything, keep every aircraft on the ground.
1. **Solve the RMP** as an LP. This gives $`z_{\text{RMP}}`$ and duals $`(\pi,\sigma,\mu)`$.
2. **Price** every aircraft: compute $`\bar c_k^{\ast}`$ and its path (§7.2).
3. **Stop or add.** If $`\bar c_k^{\ast}\ge-\varepsilon`$ for all $`k`$, stop. By §9, $`z_{\text{RMP}}-\lvert K\rvert\varepsilon\le z_{\text{LP}}\le z_{\text{RMP}}`$, so $`z_{\text{RMP}}`$ equals $`z_{\text{LP}}`$ up to $`\lvert K\rvert\varepsilon`$ (exactly, by §7.1, if $`\varepsilon=0`$). Otherwise, add the routes with $`\bar c_k^{\ast}<-\varepsilon`$ and return to step 1.

**Dual smoothing.** Set-partitioning duals jump between degenerate vertices, and that slows the loop down ("tailing-off", §6.4). The code therefore prices at smoothed multipliers (Wentges smoothing):

$$
(\tilde\pi,\tilde\sigma,\tilde\mu)=\alpha\,(\hat\pi,\hat\sigma,\hat\mu)+(1-\alpha)\,(\pi,\sigma,\mu),
$$

where $`(\pi,\sigma,\mu)`$ are the current RMP duals and the stability center $`(\hat\pi,\hat\sigma,\hat\mu)`$ is the multiplier vector with the best Lagrangian bound (§9) so far. A route priced at the smoothed multipliers enters the RMP only if its reduced cost at the RMP duals is below $`-\varepsilon`$. If no route qualifies, the same iteration prices again at the RMP duals themselves, so the stopping test of step 3 is exactly the one above. With $`\alpha=0`$ this is the plain loop; the code uses $`\alpha=0.8`$ (`config.yaml`), except in `examples/cg_by_hand.py`, which runs the plain loop.

In words: instead of jumping to the newest prices, pricing moves only part of the way from the best prices seen so far. The answer to "has column generation converged?" still comes from the RMP's own prices.

In words: the master holds a few routes and says what each flight is currently worth. Each aircraft answers with its best day at those prices. The prices change every time a route is added, until no aircraft can find a day worth more than it costs.

## 9. Lagrangian lower bound

Add the bounds $`z_f\le 1`$ and $`y_s\le b_s`$ to the LP master. The first is implied by the cover row. The second holds at every optimum because $`c^{\text{pos}}>0`$. So neither changes $`z_{\text{LP}}`$.

Now relax the cover rows with multipliers $`\pi`$ (free) and the positioning rows with $`\mu\ge 0`$, keeping the one-route rows. The relaxed problem separates by aircraft, by flight and by station:

$$
\mathcal L(\pi,\mu)=\sum_f\pi_f+\sum_s b_s\mu_s
+\sum_k\min_{r\in R_k}\Big(c_r-\sum_f a_{fr}\pi_f-\sum_s e_{sr}\mu_s\Big)
+\sum_f\min(0,\ C_f-\pi_f)
+\sum_s b_s\min(0,\ c^{\text{pos}}-\mu_s).
$$

Weak duality gives $`\mathcal L(\pi,\mu)\le z_{\text{LP}}`$ for every $`\pi`$ and every $`\mu\ge 0`$.

Evaluate it at the current RMP duals:

- the inner minimum equals $`\bar c_k^{\ast}+\sigma_k`$;
- the last two sums are 0, because $`\pi_f\le C_f`$ and $`\mu_s\le c^{\text{pos}}`$ (§7.1);
- $`\sum_f\pi_f+\sum_k\sigma_k+\sum_s b_s\mu_s=z_{\text{RMP}}`$ by strong duality on the RMP.

Hence

$$
z_{\text{RMP}}+\sum_{k\in K}\bar c_k^{\ast}\ \ \le\ \ z_{\text{LP}}\ \ \le\ \ z_{\text{RMP}} .
$$

The bound is valid at every iteration, provided every aircraft was priced exactly. It is not monotone across iterations, so keep the best value seen, $`\text{LB}_t=\max_{\le t}\mathcal L`$. Column generation can stop early once $`(z_{\text{RMP}}-\text{LB}_t)/z_{\text{RMP}}`$ falls below a tolerance. If $`z_{\text{RMP}}=0`$, then $`z_{\text{LP}}=0`$, because every cost is nonnegative, and column generation stops. Before convergence, $`z_{\text{RMP}}`$ itself is not a lower bound.

Since $`\bar c_k^{\ast}\le 0`$ (§8), the bound equals $`z_{\text{RMP}}+\sum_k\min(0,\bar c_k^{\ast})`$. The code uses this form, which stays valid when round-off in the solver's duals makes a computed $`\bar c_k^{\ast}`$ slightly positive.

With dual smoothing (§8), pricing also happens at the smoothed multipliers $`(\tilde\pi,\tilde\sigma,\tilde\mu)`$. They are convex combinations of RMP duals, so $`\tilde\pi_f\le C_f`$ and $`0\le\tilde\mu_s\le c^{\text{pos}}`$ hold for them too, and $`\mathcal L(\tilde\pi,\tilde\mu)`$ is a valid bound. There the first sum is not $`z_{\text{RMP}}`$, so the code evaluates the full expression: $`\mathcal L=\sum_f\tilde\pi_f+\sum_k\tilde\sigma_k+\sum_s b_s\tilde\mu_s+\sum_k\bar c_k^{\ast}(\tilde\pi,\tilde\sigma,\tilde\mu)`$ plus the two $`\min(0,\cdot)`$ sums, which are 0 here.

In words: each aircraft can improve the current solution by at most $`\lvert\bar c_k^{\ast}\rvert`$, so the total can drop by at most the sum.

## 10. Integer solution

**Price-and-branch.** After column generation, solve the master with $`\lambda`$ and $`z`$ binary over the generated routes only. This gives a feasible plan of cost $`z_{\text{PB}}`$.

**Bounds.** Let $`\text{LB}=z_{\text{LP}}`$ if column generation converged, and $`\text{LB}=\max_t\mathcal L_t`$ if it stopped early. The true integer optimum $`z^{\ast}`$ on the delay grid satisfies

$$
\text{LB}\ \le\ z^{\ast}\ \le\ z_{\text{PB}},
\qquad
\text{gap}=\frac{z_{\text{PB}}-\text{LB}}{z_{\text{PB}}} .
$$

If $`z_{\text{PB}}=0`$, the plan costs nothing and is optimal, because every cost is nonnegative; the gap is then reported as 0.

Price-and-branch is a heuristic. No routes are generated during branching, and the integer optimum may need a route that never looked attractive at LP prices. The gap bounds the loss.

**Exact alternative.** Branch-and-price runs column generation at every node of a branch-and-bound tree, branching on aircraft-specific arc flows.

In words: the LP gives a floor, price-and-branch gives a plan, and the gap says how far apart they can be.

## 11. Compact arc-flow model

**Variables:** $`x_{ka}\in\lbrace 0,1\rbrace`$ for each arc $`a\in A_k`$ (aircraft $`k`$ uses arc $`a`$), plus $`z_f`$ and $`y_s`$ as before. Write $`\text{in}(v)`$ and $`\text{out}(v)`$ for the arcs entering and leaving node $`v`$.

**Arc costs:** an arc into copy $`(f,d)`$ costs $`c_{ka}=c^{\text{del}} p_f d+c^{\text{swap}} [ k^0_f\neq k ]`$. Arcs into sinks cost 0.

$$
\begin{aligned}
\min\quad & \sum_{k}\sum_{a\in A_k}c_{ka}x_{ka}+\sum_f C_f z_f+\sum_s c^{\text{pos}}y_s\\
\text{s.t.}\quad
& \sum_{a\in\text{out}(\text{src}_k)}x_{ka}=1 && \forall k\\
& \sum_{a\in\text{in}(v)}x_{ka}=\sum_{a\in\text{out}(v)}x_{ka} && \forall k,\ v=(f,d)\\
& \sum_k\sum_{d\in D}\ \sum_{a\in\text{in}(f,d)}x_{ka}+z_f=1 && \forall f\\
& \sum_k\sum_{a\in\text{in}(\text{snk}_s)}x_{ka}+y_s\ge b_s && \forall s\\
& x_{ka},\ z_f\in\lbrace 0,1\rbrace,\quad y_s\ge 0
\end{aligned}
$$

**Link to the master (Dantzig–Wolfe).** For each $`k`$, let $`P_k=\lbrace x_k\ge 0:\ \text{source row and conservation rows}\rbrace`$, the polytope of unit flows from $`\text{src}_k`$ to the sinks.

- $`P_k`$ is bounded (a unit flow on a DAG), so by Minkowski–Weyl every point is a convex combination of its vertices.
- Its constraint matrix is a network matrix, hence totally unimodular, so its vertices are integral. They are exactly the incidence vectors $`\chi^r`$ of the $`\text{src}_k`$-to-sink paths, i.e. the routes.
- Substitute $`x_k=\sum_{r\in R_k}\lambda_r\chi^r`$ with $`\sum_r\lambda_r=1`$, $`\lambda\ge 0`$.
  - Arc costs add up to $`c_r`$.
  - Cover rows become $`\sum a_{fr}\lambda_r`$. Each copy on a path has exactly one in-arc, and no flight repeats by §2.3.
  - Sink rows become $`\sum e_{sr}\lambda_r`$.

The result is the master of §5. So the two LP relaxations are equivalent:

$$
z_{\text{LP}}^{\text{compact}}=z_{\text{LP}}^{\text{master}} .
$$

In words: decomposition does not tighten the bound here, because each aircraft's subproblem already has integral vertices. What it buys is scale: a small master plus fast shortest paths, instead of one MILP with a variable per aircraft and arc.

## 12. Assumptions

- **Single-day horizon.** Tomorrow enters only through $`b_s`$, and a disruption that spills past midnight is not seen.
- **Timeless sinks.** An aircraft ending at $`s`$ counts toward $`b_s`$ whatever time it lands. There are no night curfews.
- **Discrete delays.** Delays lie on the grid $`D`$, with $`D_{\max}<\bar D`$.
- **Fixed block times.** A delayed flight takes exactly as long as planned.
- **Frozen flights on schedule.** Frozen flights are assumed to have operated on schedule.
- **Closures.** The closure window is chosen so that no flight in progress at $`t_a`$ lands at $`\bar s`$ inside it (no diversions).
- **Scope.** Single fleet type. No crew, no maintenance, no passenger itineraries.
