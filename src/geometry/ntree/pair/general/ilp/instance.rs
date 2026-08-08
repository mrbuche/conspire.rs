use super::conflicts;
use std::collections::HashSet;

pub(crate) struct Instance<const D: usize> {
    cells: Vec<([i32; D], bool)>,
    /// Vertices the alignment rule has refused. Honoured only while pairing stays feasible
    /// without them, since pairing itself is not negotiable.
    forbidden: HashSet<[i32; D]>,
}

fn offset<const D: usize>(a: [i32; D], b: [i32; D]) -> [i32; D] {
    let mut offset = [0; D];
    for axis in 0..D {
        offset[axis] = b[axis] - a[axis];
    }
    offset
}

impl<const D: usize> Instance<D> {
    pub(crate) fn new(cells: Vec<([i32; D], bool)>, forbidden: HashSet<[i32; D]>) -> Self {
        Self { cells, forbidden }
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
        // Two vertices of the same cell always conflict, so no cover straddles two components
        // of the conflict graph and the components are independent problems. Solving them
        // apart keeps the search exponential in the largest component rather than in the whole
        // level, and the objective is a sum, so the pieces still compose to the optimum.
        let mut parent: Vec<usize> = (0..count).collect();
        (0..count).for_each(|i| {
            conflicts_of[i].iter().for_each(|&j| {
                let (a, b) = (find(&mut parent, i), find(&mut parent, j));
                if a != b {
                    parent[a] = b;
                }
            })
        });
        let mut component_of = vec![0; count];
        let mut local_of = vec![0; count];
        let mut of_root = vec![usize::MAX; count];
        let mut members: Vec<Vec<usize>> = Vec::new();
        for i in 0..count {
            let root = find(&mut parent, i);
            if of_root[root] == usize::MAX {
                of_root[root] = members.len();
                members.push(Vec::new());
            }
            component_of[i] = of_root[root];
            local_of[i] = members[of_root[root]].len();
            members[of_root[root]].push(i);
        }
        let mut grouped: Vec<Vec<Vec<usize>>> = vec![Vec::new(); members.len()];
        covers.iter().for_each(|cover| {
            if let Some(&first) = cover.first() {
                grouped[component_of[first]].push(cover.iter().map(|&i| local_of[i]).collect())
            }
        });
        let mut total = 0;
        let mut assignment = HashSet::new();
        for (component, cover) in members.iter().zip(grouped) {
            let valences: Vec<usize> = component.iter().map(|&i| valences[i]).collect();
            let conflicts_of: Vec<Vec<usize>> = component
                .iter()
                .map(|&i| conflicts_of[i].iter().map(|&j| local_of[j]).collect())
                .collect();
            let mut solver = Solver {
                valences: &valences,
                conflicts_of: &conflicts_of,
                covers: &cover,
                selected: vec![false; component.len()],
                excluded: component.iter().map(|&i| excluded[i]).collect(),
                best: None,
            };
            solver.branch(0);
            let (cost, selected) = solver.best.expect("no feasible assignment found");
            total += cost;
            assignment.extend(
                selected
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, chosen)| chosen.then_some(candidates[component[i]])),
            );
        }
        (assignment, total)
    }
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
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
