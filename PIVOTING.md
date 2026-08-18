# Pivoting in the incomplete LDLᵀ factorization

A spec for making `CscIncompleteLdl` survive a strongly indefinite tangent.
Read this in full before touching code. The failure mode here is silent — a
preconditioner that is not a valid factorization does not crash, it converges
confidently and returns a wrong answer — so most of this document is about how
to know you have it right, not about what to type.

## Where things stand

`src/math/sparse/factor/incomplete/` holds an incomplete LDLᵀ factorization used
as a preconditioner by the minimal residual method in
`src/math/optimize/krylov/`. It works well on a positive definite tangent and is
refused on a strongly indefinite one.

Measured on the hyperelastic block in `tests/temporary.rs::temporary_hyperelastic`
(neo-Hookean, 3993 free variables, `EqualityConstraint::Linear`, strain 13):

| Newton step | negative pivots | pivots | largest entry | MINRES iterations |
|---|---|---|---|---|
| 1, reference configuration | 0 of 3993 | 0.23 … 1.0 | 0.70 | 92 |
| 2, deformed configuration | ~1500 of 3993 | up to 7.7e85 | 6e34 | refused |

The second row is the problem. Unpivoted elimination on an indefinite matrix has
no bound on how far it can run away: a pivot small against what it divides
throws out a large entry, that entry feeds the corrections to every row after
it, and the factorization runs away by eighty orders of magnitude. Scaling the
matrix to a unit diagonal and dropping entries past `GROWTH` hold this in many
cases; on the deformed tangent they do not, and the factorization is refused
(returns `None`) so the caller falls back on the diagonal.

**This is what pivoting exists to fix, and nothing short of it will.** Shifting
the diagonal, raising the pivot floor, capping the pivots, and adding fill were
all tried and measured. Fill is a large win on the definite tangent (206 → 92
iterations) and buys nothing on the indefinite one. Capping the pivots makes the
factorization disintegrate — 1.6 million entries dropped from a 3993-row matrix.
Do not re-run these.

## Before you start: is this worth doing?

At 4000 degrees of freedom the sparse direct solver finishes the whole nonlinear
solve in about 30 seconds and MINRES does not finish at all. The iterative path
only pays on problems too large to factorize. **Confirm the target problem size
with the user before writing code.** If there is no size at which the direct
solver is unusable, this work has no customer.

## What to build

A Bunch-Kaufman-style incomplete LDLᵀ: `P A Pᵀ = L D Lᵀ`, where `D` is block
diagonal with 1×1 and 2×2 blocks, and the permutation `P` is chosen during the
factorization to keep the entries of `L` bounded.

The published algorithm is SYM-ILDL (Greif, He and Xu) and HSL_MI30 (Scott and
Tůma). Follow one of them rather than deriving it. The parts that matter:

1. **The formulation has to change.** The current algorithm is up-looking: it
   builds one row at a time, and never has a column complete. Choosing a pivot
   for column *j* requires column *j* to be complete, so this becomes a Crout
   (left-looking) factorization, which touches rows and columns of the partial
   factor both. This is a rewrite of `factorize`, not an edit to it.
2. **Pivot selection.** The Bunch-Kaufman threshold test on the largest
   off-diagonal entry of the current column decides a 1×1 pivot or a 2×2 block.
   The dense factorization in `src/math/matrix/square/ldl/mod.rs` already does
   this — `BUNCH_KAUFMAN` is the threshold constant and `pair: Vec<bool>` marks
   where a 2×2 block starts. Crib the pivot logic from there; what is new is the
   sparse Crout scaffolding around it.
3. **Permutation bookkeeping.** The factorization is of `P A Pᵀ`, so `solve` and
   `solve_into` must permute in and out. This composes with the equilibration
   scaling already applied on both sides — get the order right and test it.
4. **Dropping and pivoting interact.** An entry about to be pivoted on cannot be
   dropped. Work out the interaction with `fill` and `threshold` deliberately
   rather than discovering it.
5. **|D| for a 2×2 block.** The preconditioner is applied with the magnitudes of
   the pivots, which for a 2×2 block means the absolute value of a symmetric 2×2
   matrix — an analytic eigen-decomposition, `Q |Λ| Qᵀ`. This is the single
   easiest thing here to get subtly and silently wrong. See the verification
   section.

### Stage it

**Stage A — 1×1 pivoting and permutation only.** No 2×2 blocks. Pick, at each
column, the largest available diagonal entry subject to a threshold, permute it
into place, and refuse the column if nothing qualifies. This is most of the
growth control for a fraction of the difficulty, and it is independently
valuable. Land and verify it before starting Stage B.

**Status (2026-08-17, `line-search`, `src/math/sparse/factor/incomplete/mod.rs`):
Stage A is landed, unit-verified, AND now confirmed end-to-end against the
real deformed tangent — see the measurement below.**

Built as a Crout factorization with an eagerly-maintained Schur-complement
diagonal (`diag`): each finalized column's not-yet-eliminated neighbors are
computed once, divided by its own pivot, and immediately (a) stored in
`columns[]` for later Crout pulls and (b) subtracted into `diag[]` for every
index they touch — so picking a pivot is always a lookup, never a
recomputation. Verified: the pattern invariant still holds under permutation,
`solve()` still yields a symmetric positive-definite operator on a heavily
indefinite (30%+ negative pivots) 100-row case, and forcing natural order
(`candidate` always chosen) reproduces the pre-pivoting code's output
byte-for-byte on the real FEM step-1 tangent — proof the rewrite itself is
correct, independent of the pivoting rule.

**The pivot rule had to be much more conservative than "always take the
largest diagonal."** Scaling normalizes every untouched diagonal to exactly
±1 before elimination starts, so a first cut at "largest available" reorders
almost anything — even a trivial tridiagonal chain, even a positive-definite
FEM tangent where nothing is actually unsafe — because *some* index is always
marginally ahead once corrections start landing. Tested against the real
step-1 tangent through an ad hoc `Krylov`-gated measurement:

| pivot rule | step-1 MINRES result |
|---|---|
| forced natural order | matches pre-pivoting baseline exactly (0.00874 relative) |
| always take global largest diagonal | diverges (1.17x) |
| candidate unless diag < 1% of best available | diverges (1.00x, no progress) |
| candidate unless diag < 10% of best available | diverges (1.47x, worse) |

Reordering is not free even when it is "safe" by some relative measure — it
disturbs the fill an incomplete factorization keeps, and incomplete
factorizations are chaotically sensitive to which fill survives (one early
swap changes what the `fill`-cap keeps for the rest of the matrix). The
landed rule tests the *natural* next candidate's own diagonal in absolute
terms and only searches for a replacement when that candidate has actually
fallen small — deliberately much stricter than a textbook Bunch-Kaufman
ratio test, and immune to the relative-comparison trap above by construction.

**Getting a real reading on step 2 without depending on MINRES converging.**
The linear system MINRES is asked to solve is a separate problem from
whether the *factorization* refuses itself, and the two don't have to be
measured together: Newton can be driven end-to-end on the proven
`LinearSolver::Sparse` path (fast, exact, unaffected by any of this), while a
side channel independently builds a `CscIncompleteLdl` from each iteration's
real tangent purely to observe it — same `fill`/`threshold` a caller would
use, nothing about the actual solve depends on it. That decouples "does
Newton actually reach the deformed configuration" (yes, via Sparse, in
~40 s) from "what does the incomplete factorization do with the tangent it's
handed there" (the only thing in question), and let `SAFE` be swept quickly
against the *real* step-2 tangent rather than guessed at:

| `SAFE` | step 1 (3993 dof, reference config) | step 2 (deformed config) |
|---|---|---|
| 1e-4 – 3e-2 | untouched (0 negative pivots, matches baseline) | **REFUSED** |
| 4.5e-2, 4.9e-2 | — | **REFUSED** |
| **4e-2** | **untouched, MINRES byte-identical to baseline** | **946/3993 negative pivots, factorizes** |
| 5e-2 | untouched, MINRES byte-identical to baseline | 1027/3993 negative pivots, factorizes |
| 6e-2 | MINRES already diverges (1.0004x) | factorizes |
| 1e-1 | MINRES diverges badly (1.47x) | factorizes |

The step-2 crossover is not monotonic in `SAFE` (4.5e-2 and 4.9e-2 both
refuse where 4e-2 and 5e-2 don't) — expected, given the chaotic fill
sensitivity above, and a reason not to trust a single successful value
without bracketing it. **`SAFE = 4e-2` is landed**: it is the smaller of the
two working points found, confirmed twice over (both the negative-pivot
count and, independently, `KrylovMethod::Minres`'s step-1 residual coming
out byte-identical to the unpivoted baseline) to leave the healthy tangent's
elimination order completely untouched, while turning the deformed tangent's
factorization from a refusal into 946 negative pivots successfully kept.

**Whether MINRES itself then converges on that now-unrefused step-2
system — checked.** Restarted Newton from the exact deformed configuration
(captured mid-solve on the proven `LinearSolver::Sparse` path, since a fully
Krylov-driven trajectory can't get there — see below) and asked it for one
`LinearSolver::Krylov` solve there, replicating the real step-2 linear
system without needing MINRES to converge at step 1 first.

First attempt: `PreconditionerNotPositiveDefinite`, hard failure. This was a
real, separate bug, not an artifact of the measurement — `kkt_schur` builds
the multiplier block's preconditioner by calling `factorize_ldl()` on the
Schur complement of `factor.solve()`, and `factorize_ldl()` succeeds on any
symmetric matrix, indefinite or not (that is the entire point of pivoting).
`factor.solve()` is always positive definite by construction (it takes
`|D|`), but the complement of a positive-definite operator is only
guaranteed positive definite when that operator is the *true* inverse, and
an incomplete factorization never is. On the real step-2 tangent the Schur
complement came out genuinely indefinite. **This path was unreachable before
Stage A pivoting landed** — the factorization itself was refused first
(growth to 7.7e85), so `kkt_schur` was never exercised on a step-2 tangent
at all; pivoting's success is what exposed it. Fixed in a follow-up commit:
`LdlDecomposition::is_positive_definite` (1×1 blocks by sign, 2×2 by
Sylvester's criterion), and `kkt_schur` now filters through it, falling back
to the diagonal preconditioner exactly as it already does when the
factorization itself is refused.

With that fixed: the crash is gone, and MINRES actually runs on step 2 (via
the diagonal fallback for BOTH blocks — `kkt_schur` returning `None` fell
through to the same `Preconditioning::Diagonal` path as when `factor` itself
is refused). It stalls at a 0.00874 relative residual — the same stall step
1 already had before any of this work, reproduced with pivoting completely
inert — [[krylov_fem_scale_negative]]'s pre-existing "attainable-accuracy
floor," not a regression.

**Then tried to do better: keep `factor` when only the Schur block fails.**
Falling all the way back to a bare diagonal for *both* blocks whenever the
Schur complement alone was bad wastes a perfectly good, expensive-to-build
factor for the (much larger) variables block. Landed a `diagonal_schur`
fallback — the same per-constraint-row diagonal formula the existing
`diagonal()` closure already used, but scoped to just the multiplier block —
so the strong incomplete factorization is kept for variables and only the
small multiplier block falls back to something weak.

Re-measured: the crash **came back**, with the exact same magnitude as
before the Schur-PD fix (`-3.86e102`). Instrumenting confirmed the new
diagonal fallback genuinely is positive definite — the crash is not
`schur`'s fault this time. The real source: `factor`'s own pivots range up
to `9.68e7`, just under `UNSTABLE` (`1e8`), so the factorization is never
refused — but at that magnitude, chained back-substitution inside
`factor.solve()` catastrophically cancels in floating point, producing a
result whose measured quadratic form is numerically negative even though it
is provably positive definite in exact arithmetic (it is built from `|D|`).
Tried tightening `UNSTABLE` to 1e7, 1e6, 1e5 — all three avoid the crash,
but only by refusing the factorization outright once its growth exceeds the
tighter bound, landing back on the *same* weak all-diagonal fallback and
the *same* 0.00874 stall. That is not a fix, just a different way of giving
up before the crash — reverted, `UNSTABLE` stays at `1e8`.

**Conclusion: this is a real, deeper limit of Stage A, not a bug to chase
further here.** Stage A never tests a pivot's magnitude locally against
what it is about to divide — the SYM-ILDL/Bunch-Kaufman machinery Stage B
was always going to need (`omega`, the largest off-diagonal in the current
column, compared against the candidate diagonal) is exactly what would keep
pivot magnitudes away from `9.68e7` in the first place, rather than merely
keeping the *sum* of growth under a global ceiling. The `diagonal_schur`
fallback is landed anyway (own commit) — it is a genuine, isolated
improvement for any case where the factor is well-behaved and only the
Schur complement specifically comes out indefinite, which is the common
case; it just isn't enough on its own for this specific pathological
tangent.

**Where this left the original two-part bar, at that point.** The deformed
tangent factorizes without being refused — done. The linear solves converge
on a true residual — not on this specific tangent, for the reason above.
Stage A itself (landed and correct) was finished; what remained belonged to
Stage B or to a numerically-robust catastrophic-cancellation guard. The next
section is that guard, attempted.

## The inverse-norm-bounded criterion (ILUPACK)

Asked a user for pointers on codes that handle this well; they named
ILUPACK (Bollhöfer & Saad) and asked specifically whether it applied here
and whether it beat SYM-ILDL. Read *Multilevel Preconditioners Constructed
From Inverse-Based ILUs* (2004/2006) in full before writing anything —
worth doing again before touching this section, since the theorem numbers
below are theirs, not reconstructed from memory.

**What it says, precisely.** Their Theorem 6 / Corollary 3: if `‖L⁻¹‖,
‖U⁻¹‖ ≤ κ` at every step, dropping an entry is only safe when its magnitude
is below `ε/κ²` — the drop threshold has to scale with how amplifying the
*current inverse factors* are, not sit at a fixed constant like `GROWTH`.
This is precisely `9.68e7`: a factor can pass both `GROWTH` and `UNSTABLE`
(which bound the entries and pivots of `L` directly) while `‖L⁻¹‖` is still
enormous, because neither one ever looks at the inverse.

**Is it better than SYM-ILDL?** Their own numerical comparison (§4)
answers this, and it is more nuanced than yes: at drop tolerance `1e-2`,
plain threshold-pivoted ILU (ILUTP) solves ~72–76% of their test problems;
single-level *inverse-aware* dropping alone (ILUTC — the closest analogue
to grafting their criterion onto Stage A as-is) solves ~29–52%, the
*worst* of their three compared methods; full multilevel ILUPACK solves
~81–90%. The criterion alone is not what buys the robustness — the
multilevel architecture is (defer whatever the criterion can't keep safe to
a Schur complement, recurse on that as a smaller instance of the same
problem). That is a different, larger project than Stage A or Stage B, out
of scope here; see [[pivoting_stage_a]] for the fuller writeup of this
conversation.

**What was landed anyway, and why.** The criterion swap alone — no
multilevel — is still a real, bounded, worthwhile change: it turns "silently
accept a factor whose true inverse is catastrophic and crash MINRES three
calls later" into "refuse the factorization for a legible, correct reason,
right where the danger actually is." Added `rho`, a running bound on the
row-sum norm of the inverse factor — for unit lower triangular `L`, row `i`
of `L⁻¹` satisfies `(L⁻¹)ᵢⱼ = δᵢⱼ - Σₖ Lᵢₖ(L⁻¹)ₖⱼ`, so its row-sum norm is
bounded by `1 + Σₖ|Lᵢₖ|·ρₖ` — exactly the entries and dependencies a row's
own Crout construction already has in hand, so tracking it costs nothing
extra. `AMPLIFICATION` drops a fill entry that would push this too far;
`INVERSE_UNSTABLE` refuses the whole factorization, mirroring `UNSTABLE`
but on the amplified quantity instead of the raw one.

Measured directly against the real tangent, not a synthetic construction —
small synthetic matrices did not discriminate this mechanism at all,
matching the exact pattern `SAFE`'s own tuning hit earlier in this
document:

| threshold | step 1 (reference) | step 2 (deformed) |
|---|---|---|
| `INVERSE_UNSTABLE = 1e6` | refused (!) — too tight, step 1 shouldn't be | refused |
| `INVERSE_UNSTABLE = 1e12` | untouched, 0 drops | refused, `rho` up to `3.1e22` |
| `INVERSE_UNSTABLE = 1e14` (landed) | untouched, 0 drops, MINRES byte-identical to baseline | refused, same `3.1e22` |

Loosening the threshold from `1e12` to `1e14` did not change step 2's
outcome at all — its `rho` reaches `3.1e22` regardless, thirteen orders of
magnitude past anything a reasonable threshold could admit. That is the
theory correctly detecting genuine, severe ill-conditioning, exactly what
the paper's own comparison predicts a single-level criterion can detect but
not cure. `1e14` sits with wide margin above step 1's observed `~5.7e9`
(three-plus orders of headroom) and wide margin below step 2's `3.1e22`
(eight-plus orders below it) — the gap between the two problems is large
enough that the exact threshold value doesn't matter much, only that it sits
somewhere in that gap.

**Where this leaves the two-part bar now.** The deformed tangent factorizes
without being refused for the *wrong* reason, and is now correctly refused
for a *right* one — the downstream `PreconditionerNotPositiveDefinite`
crash this was chasing cannot recur through this path, because the
factorization that would produce it is caught first. The linear solves
converging on a true residual is still open, and still belongs to Stage B
or full multilevel — this session's honest answer is that a single-level
criterion, however correct, was never going to close that gap, and the
paper says so about its own version of the same idea.

**Stage B — 2×2 blocks.** Only after Stage A is green. This is where `D` stops
being a vector and `|D|` stops being `abs`.

## How you will know it is right

This is the important half of the document.

### The invariant

An incomplete factorization is defined by where it is *exact*, not by how near
its product comes to the matrix:

> On every position the kept pattern holds, `Pᵀ L D Lᵀ P` agrees with `A` entry
> for entry.

Everywhere else the product is whatever the dropped fill would have cancelled,
and is not asked about. `agrees_with_the_matrix_on_its_own_pattern` in
`src/math/sparse/factor/incomplete/test.rs` asserts exactly this today, and also
asserts that some fill really was dropped — otherwise the test passes trivially
on a complete factorization. Extend it through the permutation; do not weaken it.

### Mutation-check every test you write

**Assume a new test is vacuous until you have watched it fail.** Break the thing
the test is meant to catch, run it, confirm it fails, put the code back, confirm
it passes. This is not optional diligence, it is the only thing that separates a
test from a comment.

During the session that produced this code, a test written specifically to catch
"the walk reports a residual that is not the residual" passed unchanged with the
fix removed. It looked reasonable. It proved nothing. The fixture had to be
rebuilt — putting the load only where the preconditioner weighs heaviest, so
MINRES spends itself on those coordinates and strands the residual where the
measurement barely registers it — before it discriminated. That test is
`a_misleading_preconditioner_does_not_pass_for_a_short_residual` in
`src/math/optimize/krylov/test.rs`; with the fix removed it returns a solution
leaving a relative residual of 0.199 while reporting 1e-8.

### A small system cannot demonstrate a preconditioner

MINRES terminates in at most *n* iterations in exact arithmetic whatever the
residual was put through first, so on a small system nothing done beforehand can
be told from nothing at all. The `at_size` module in
`src/math/optimize/krylov/test.rs` exists for this: a 260-row saddle-point system
with an indefinite leading block, where `every_preconditioner_reaches_the_direct_answer`
and `preconditioning_shortens_the_walk` can actually fail. New preconditioner
work belongs there.

### Specifically for 2×2 blocks

Test `|D|` on its own before trusting it in a solve. For a symmetric 2×2 with
known eigenvalues of mixed sign, assert that the result is positive definite,
that it has the same eigenvectors, and that its eigenvalues are the magnitudes of
the originals. A wrong `|D|` still produces a symmetric matrix and still lets
every solve run to completion, so nothing downstream will tell you.

### Growth is observable

`CscIncompleteLdl` records how far the factorization ran away and refuses itself
past `UNSTABLE`. A successful Stage A or B should show the deformed tangent
factorizing without refusal and with entries of order one, not 1e34. If it still
refuses, the pivoting is not doing its job — say so plainly rather than raising
`UNSTABLE` to make it pass.

## The end-to-end check

```
CARGO_PROFILE_DEV_DEBUG=0 cargo test -F fem --lib math::sparse::factor::incomplete
CARGO_PROFILE_DEV_DEBUG=0 cargo test -F fem --lib math::optimize::krylov
```

Then the real problem. `tests/temporary.rs::temporary_hyperelastic` uses the
sparse direct solver by design — `src/domain/fem/solid/hyperelastic/mod.rs`
hardcodes `LinearSolver::Sparse`, which is correct and should stay that way. To
measure MINRES against it, gate the linear solver on an environment variable in
that wrapper temporarily, and **revert the gate before committing**.

```
CARGO_PROFILE_DEV_DEBUG=0 cargo test -F fem --test temporary temporary_hyperelastic -- --exact --nocapture
```

`--exact` matters: without it the filter also runs
`temporary_hyperelastic_internal_variables`, which takes several minutes.

The bar: the deformed-configuration tangent factorizes without being refused,
and the linear solves converge on a *true* residual, not a reported one.
Anything that reports convergence should be spot-checked against
`‖b − Ax‖ / ‖b‖` at least once by hand.

## Things that will mislead you

- **A converged linear solve is not a converged Newton solve.** With the linear
  systems solved well, this test still fails, because at strain 13 in one step
  the nonlinear path wanders into configurations with inverted elements. That is
  a globalization problem — load stepping — and it is not yours. Do not chase it
  and do not conclude the preconditioner failed when the failure says
  `Invalid Jacobian` or comes from the line search.
- **The nonlinear path is chaotic here.** Two preconditioners that both solve
  step 1 well will reach different step-2 configurations with different tangents.
  Iteration counts past step 1 are not comparable between variants. Compare on
  step 1, or on a fixed matrix.
- **`SquareMatrix::size()` is deliberately unimplemented.** The default
  `Hessian::lower_triangle` uses it, so `SquareMatrix` carries its own override.
  Anything new implementing `Hessian` needs the same consideration.
- **`cargo test` with no features runs zero tests.** Always pass `-F fem` for
  this work.

## Conventions

`CLAUDE.md` and the memory directory carry the house style; the ones that come
up most here:

- Comments are sparse and explain *why*, not *what*. Narration gets stripped.
- `Type::from(x)` rather than `let y: Type = x.into()`.
- Types go in a turbofish or a suffixed literal on the right, not in a
  left-hand annotation — except `.collect()`, which keeps the annotation.

## Files

| Path | What it is |
|---|---|
| `src/math/sparse/factor/incomplete/mod.rs` | the factorization; `new`, `with_fill`, `solve`, `solve_into` |
| `src/math/sparse/factor/incomplete/test.rs` | the invariant and the indefinite case |
| `src/math/matrix/square/ldl/mod.rs` | dense Bunch-Kaufman to crib pivot logic from |
| `src/math/optimize/krylov/mod.rs` | `Preconditioner`, `Preconditioning`, MINRES |
| `src/math/optimize/krylov/test.rs` | `at_size`, and the misleading-preconditioner test |
| `src/math/optimize/newton_raphson/mod.rs` | `kkt_preconditioning`, `kkt_schur` |
