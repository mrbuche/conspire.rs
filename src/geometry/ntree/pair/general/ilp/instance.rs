use super::conflicts;
use std::collections::HashSet;

pub(crate) struct Instance<const D: usize> {
    cells: Vec<([i32; D], bool)>,
}

fn offset<const D: usize>(a: [i32; D], b: [i32; D]) -> [i32; D] {
    let mut offset = [0; D];
    for axis in 0..D {
        offset[axis] = b[axis] - a[axis];
    }
    offset
}

impl<const D: usize> Instance<D> {
    pub(crate) fn new(cells: Vec<([i32; D], bool)>) -> Self {
        Self { cells }
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
    fn valence(&self, vertex: [i32; D]) -> usize {
        self.cells
            .iter()
            .filter(|(cell, _)| Self::vertices_of(*cell).contains(&vertex))
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
        let vertices: Vec<_> = assignment.iter().collect();
        for (i, vertex_i) in vertices.iter().enumerate() {
            for vertex_j in &vertices[i + 1..] {
                if conflicts(offset(**vertex_i, **vertex_j)) {
                    return false;
                }
            }
        }
        true
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
        let valences: Vec<usize> = candidates
            .iter()
            .map(|&vertex| self.valence(vertex))
            .collect();
        let conflicts_of: Vec<Vec<usize>> = (0..count)
            .map(|i| {
                (0..count)
                    .filter(|&j| j != i && conflicts(offset(candidates[i], candidates[j])))
                    .collect()
            })
            .collect();
        let covers: Vec<Vec<usize>> = self
            .cells
            .iter()
            .filter(|(_, required)| *required)
            .map(|(cell, _)| {
                let vertices = Self::vertices_of(*cell);
                (0..count)
                    .filter(|&i| vertices.contains(&candidates[i]))
                    .collect()
            })
            .collect();
        let mut solver = Solver {
            valences: &valences,
            conflicts_of: &conflicts_of,
            covers: &covers,
            selected: vec![false; count],
            excluded: vec![false; count],
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
    selected: Vec<bool>,
    excluded: Vec<bool>,
    best: Option<(usize, Vec<bool>)>,
}

impl Solver<'_> {
    fn branch(&mut self, cost: usize) {
        if let Some((best_cost, _)) = &self.best
            && cost >= *best_cost
        {
            return;
        }
        let uncovered = self
            .covers
            .iter()
            .filter(|cover| !cover.iter().any(|&i| self.selected[i]))
            .min_by_key(|cover| cover.iter().filter(|&&i| !self.excluded[i]).count());
        let Some(cover) = uncovered else {
            self.best = Some((cost, self.selected.clone()));
            return;
        };
        let options: Vec<usize> = cover
            .iter()
            .copied()
            .filter(|&i| !self.excluded[i])
            .collect();
        for i in options {
            self.selected[i] = true;
            let newly_excluded: Vec<usize> = self.conflicts_of[i]
                .iter()
                .copied()
                .filter(|&j| !self.excluded[j])
                .collect();
            for &j in &newly_excluded {
                self.excluded[j] = true;
            }
            self.branch(cost + self.valences[i]);
            self.selected[i] = false;
            for &j in &newly_excluded {
                self.excluded[j] = false;
            }
        }
    }
}
