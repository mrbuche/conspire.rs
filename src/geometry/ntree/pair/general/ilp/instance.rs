use super::conflicts;
use std::collections::HashSet;

pub(crate) struct Instance<const D: usize> {
    cells: Vec<([i32; D], bool)>,
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
                let mut offset = [0; D];
                for axis in 0..D {
                    offset[axis] = vertex_j[axis] - vertex_i[axis];
                }
                if conflicts(offset) {
                    return false;
                }
            }
        }
        true
    }
    pub(crate) fn cost(&self, assignment: &HashSet<[i32; D]>) -> usize {
        assignment.iter().map(|&vertex| self.valence(vertex)).sum()
    }
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
}
