use super::conflicts;
use crate::math::{FxHashMap, FxHashSet};
use std::{array::from_fn, collections::HashSet};

pub(crate) struct Instance<const D: usize> {
    cells: Vec<([i32; D], bool)>,
    /// The same cells by position, so that the neighbourhood of a vertex can be asked for
    /// rather than found by scanning every cell.
    positions: FxHashSet<[i32; D]>,
    /// Vertices the alignment rule has refused. Honoured only while pairing stays feasible
    /// without them, since pairing itself is not negotiable.
    forbidden: HashSet<[i32; D]>,
}

impl<const D: usize> Instance<D> {
    pub(crate) fn new(cells: Vec<([i32; D], bool)>, forbidden: HashSet<[i32; D]>) -> Self {
        let positions = cells.iter().map(|&(cell, _)| cell).collect();
        Self {
            cells,
            positions,
            forbidden,
        }
    }
    fn vertices_of(cell: [i32; D]) -> Vec<[i32; D]> {
        let mut vertices = vec![cell];
        for axis in 0..D {
            vertices = vertices
                .into_iter()
                .flat_map(|vertex| {
                    let mut shifted = vertex;
                    shifted[axis] += 1;
                    [vertex, shifted]
                })
                .collect();
        }
        vertices
    }
    fn candidates(&self) -> Vec<[i32; D]> {
        let mut candidates: Vec<_> = self
            .cells
            .iter()
            .filter(|(_, required)| *required)
            .flat_map(|(cell, _)| Self::vertices_of(*cell))
            .collect();
        candidates.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        candidates.dedup();
        candidates
    }
    /// A vertex is a corner of exactly the `2^D` cells reached by stepping back from it along
    /// any subset of the axes, so ask for those rather than scanning every cell for it.
    fn valence(&self, vertex: [i32; D]) -> usize {
        (0..1usize << D)
            .filter(|bits| {
                let cell: [i32; D] = from_fn(|axis| vertex[axis] - ((bits >> axis) & 1) as i32);
                self.positions.contains(&cell)
            })
            .count()
    }
    pub(crate) fn feasible(&self, assignment: &HashSet<[i32; D]>) -> bool {
        if self.cells.iter().any(|(cell, required)| {
            *required
                && !Self::vertices_of(*cell)
                    .iter()
                    .any(|vertex| assignment.contains(vertex))
        }) {
            return false;
        }
        // Conflict needs every axis within two, so only the `5^D` neighbourhood around each
        // vertex can possibly conflict with it - the same walk `solve` uses, instead of the
        // O(n^2) all-pairs check this replaces. Each conflicting pair is found from both ends,
        // which is redundant but still linear in the assignment size.
        assignment.iter().all(|&vertex| {
            (0..5usize.pow(D as u32)).all(|code| {
                let step: [i32; D] =
                    from_fn(|axis| (code / 5usize.pow(axis as u32) % 5) as i32 - 2);
                if step.iter().all(|&s| s == 0) || !conflicts(step) {
                    return true;
                }
                let other: [i32; D] = from_fn(|axis| vertex[axis] + step[axis]);
                !assignment.contains(&other)
            })
        })
    }
    #[cfg(test)]
    pub(crate) fn cost(&self, assignment: &HashSet<[i32; D]>) -> usize {
        assignment.iter().map(|&vertex| self.valence(vertex)).sum()
    }
    #[cfg(test)]
    pub(crate) fn solve_bruteforce(&self) -> (HashSet<[i32; D]>, usize) {
        let candidates = self.candidates();
        let count = candidates.len();
        assert!(count < 32, "brute force is exponential in candidate count");
        (0..1u32 << count)
            .filter_map(|mask| {
                let assignment: HashSet<_> = (0..count)
                    .filter(|bit| mask & (1 << bit) != 0)
                    .map(|bit| candidates[bit])
                    .collect();
                self.feasible(&assignment)
                    .then(|| (self.cost(&assignment), assignment))
            })
            .min_by_key(|(cost, _)| *cost)
            .map(|(cost, assignment)| (assignment, cost))
            .expect("no feasible assignment found")
    }
    pub(crate) fn solve(&self) -> (HashSet<[i32; D]>, usize) {
        let candidates = self.candidates();
        let count = candidates.len();
        // `candidates` is sorted, so a binary search per lookup would do, but a hash lookup is
        // O(1) rather than O(log count) and this runs ~5^D times per candidate plus 2^D times
        // per required cell - both dominated by the probe, not by comparisons.
        let index: FxHashMap<[i32; D], usize> = candidates
            .iter()
            .enumerate()
            .map(|(i, &vertex)| (vertex, i))
            .collect();
        let valences: Vec<usize> = candidates
            .iter()
            .map(|&vertex| self.valence(vertex))
            .collect();
        // Conflict needs every axis within two, so a vertex can only conflict with the `5^D`
        // around it. Walking those beats comparing every pair, which is quadratic in a set that
        // grows with the level. Which of the `5^D` offsets actually conflict never depends on
        // the candidate, so that filter runs once here rather than once per candidate.
        let conflict_offsets: Vec<[i32; D]> = (0..5usize.pow(D as u32))
            .filter_map(|code| {
                let offset: [i32; D] =
                    from_fn(|axis| (code / 5usize.pow(axis as u32) % 5) as i32 - 2);
                (offset.iter().any(|&step| step != 0) && conflicts(offset)).then_some(offset)
            })
            .collect();
        let conflicts_of: Vec<Vec<usize>> = candidates
            .iter()
            .map(|&vertex| {
                let mut conflicting: Vec<usize> = conflict_offsets
                    .iter()
                    .filter_map(|offset| {
                        let other: [i32; D] = from_fn(|axis| vertex[axis] + offset[axis]);
                        index.get(&other).copied()
                    })
                    .collect();
                conflicting.sort_unstable();
                conflicting
            })
            .collect();
        let covers: Vec<Vec<usize>> = self
            .cells
            .iter()
            .filter(|(_, required)| *required)
            .map(|(cell, _)| {
                let mut cover: Vec<usize> = Self::vertices_of(*cell)
                    .iter()
                    .filter_map(|vertex| index.get(vertex).copied())
                    .collect();
                cover.sort_unstable();
                cover
            })
            .collect();
        // Honour as much of the alignment rule as pairing allows: a refusal is withdrawn only
        // where it would leave a cell with nothing to cover it, and only for that cell.
        let mut excluded: Vec<bool> = candidates
            .iter()
            .map(|vertex| self.forbidden.contains(vertex))
            .collect();
        while let Some(cover) = covers
            .iter()
            .find(|cover| !cover.is_empty() && cover.iter().all(|&i| excluded[i]))
        {
            excluded[cover[0]] = false;
        }
        // Every cover of the same required cell's candidates always conflicts pairwise (any two
        // vertices of one cell are within the doubled-grid spacing of each other), so in
        // principle covers could split into independent components with no candidate in common.
        // Measured on real adaptive-mesh geometry (2026-09, bone STL benchmark): they never do -
        // every level solve is one component - so a union-find here to isolate them costs real
        // time (~3% of a solve) to confirm something already true, not to avoid exponential
        // search. Solve the whole candidate set directly instead.
        let cover_of: Vec<Vec<usize>> = {
            let mut cover_of = vec![Vec::new(); count];
            for (c, cover) in covers.iter().enumerate() {
                for &i in cover {
                    cover_of[i].push(c);
                }
            }
            cover_of
        };
        let covered: Vec<u32> = vec![0; covers.len()];
        let viable: Vec<u32> = covers
            .iter()
            .map(|cover| cover.iter().filter(|&&i| !excluded[i]).count() as u32)
            .collect();
        let mut solver = Solver {
            valences: &valences,
            conflicts_of: &conflicts_of,
            covers: &covers,
            cover_of: &cover_of,
            selected: vec![false; count],
            excluded_count: excluded.iter().map(|&b| b as u32).collect(),
            covered,
            viable,
            best: None,
        };
        solver.branch(0);
        let (cost, selected) = solver.best.expect("no feasible assignment found");
        let assignment = selected
            .into_iter()
            .enumerate()
            .filter_map(|(i, chosen)| chosen.then_some(candidates[i]))
            .collect();
        (assignment, cost)
    }
}

struct Solver<'a> {
    valences: &'a [usize],
    conflicts_of: &'a [Vec<usize>],
    covers: &'a [Vec<usize>],
    /// Which covers each candidate belongs to - `covers` inverted, so selecting or excluding a
    /// candidate can update exactly the covers it touches instead of rescanning every cover.
    cover_of: &'a [Vec<usize>],
    selected: Vec<bool>,
    /// How many currently-selected candidates (plus, for a candidate the alignment rule refused
    /// outright, one permanent count of its own) exclude this one. `> 0` is `excluded`. Tracking
    /// a count instead of a bool needs no separate "newly excluded" set to undo on backtrack:
    /// incrementing and decrementing around a selection is self-inverse regardless of who else
    /// currently excludes the same candidate, so it also needs no allocation per branch.
    excluded_count: Vec<u32>,
    /// Per cover: how many of its candidates are currently selected. `> 0` is "covered". Finding
    /// the most-constrained uncovered cover used to rescan every cover's members from scratch at
    /// every branch node; maintaining this (and `viable` below) incrementally turns that into an
    /// O(1) check per cover instead of O(cover size), which is what most of that scan cost was.
    covered: Vec<u32>,
    /// Per cover: how many of its candidates are not excluded right now.
    viable: Vec<u32>,
    best: Option<(usize, Vec<bool>)>,
}

impl Solver<'_> {
    fn select(&mut self, i: usize) {
        self.selected[i] = true;
        for &c in &self.cover_of[i] {
            self.covered[c] += 1;
        }
        for &j in &self.conflicts_of[i] {
            self.excluded_count[j] += 1;
            if self.excluded_count[j] == 1 {
                for &c in &self.cover_of[j] {
                    self.viable[c] -= 1;
                }
            }
        }
    }
    fn deselect(&mut self, i: usize) {
        for &j in &self.conflicts_of[i] {
            if self.excluded_count[j] == 1 {
                for &c in &self.cover_of[j] {
                    self.viable[c] += 1;
                }
            }
            self.excluded_count[j] -= 1;
        }
        for &c in &self.cover_of[i] {
            self.covered[c] -= 1;
        }
        self.selected[i] = false;
    }
    fn branch(&mut self, cost: usize) {
        if let Some((best_cost, _)) = &self.best
            && cost >= *best_cost
        {
            return;
        }
        let uncovered = (0..self.covers.len())
            .filter(|&c| self.covered[c] == 0)
            .min_by_key(|&c| self.viable[c]);
        let Some(c) = uncovered else {
            self.best = Some((cost, self.selected.clone()));
            return;
        };
        for &i in &self.covers[c] {
            if self.excluded_count[i] > 0 {
                continue;
            }
            self.select(i);
            self.branch(cost + self.valences[i]);
            self.deselect(i);
        }
    }
}
