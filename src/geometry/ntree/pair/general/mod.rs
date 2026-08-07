mod ilp;
#[cfg(test)]
mod test;

use crate::geometry::ntree::{
    Orthotree, balance::Balancing, dual::leaf_containing, node::split::Split,
};
use ilp::Instance;
use std::{
    array::from_fn,
    collections::{BTreeSet, HashSet},
    ops::Add,
};

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    pub(super) fn pair_generalized(&mut self) -> Result<bool, &'static str> {
        let lengths: BTreeSet<usize> = (0..self.len())
            .map(|index| self[U::from(index)].length.into())
            .collect();
        let lengths: Vec<usize> = lengths.into_iter().collect();
        self.pairing_vertices.clear();
        let mut paired = true;
        for window in lengths.windows(2) {
            let (fine, coarse) = (window[0], window[1]);
            if !self.pair_level(coarse, fine)? {
                paired = false;
            }
        }
        Ok(paired)
    }
    fn pair_level(&mut self, coarse: usize, fine: usize) -> Result<bool, &'static str> {
        if coarse != 2 * fine {
            return Err(
                "generalized pairing requires adjacent levels to differ by a factor of two",
            );
        }
        let coarse_nodes: Vec<(U, [i32; D], bool)> = (0..self.len())
            .filter(|&index| self[U::from(index)].length.into() == coarse)
            .map(|index| {
                let idx = U::from(index);
                let required = self[idx].is_tree();
                let corner = self[idx]
                    .corner
                    .map(|coordinate| (coordinate.into() / coarse) as i32);
                (idx, corner, required)
            })
            .collect();
        // Pitzalis et al. S5.2 and Fig. 9: a cluster and a region two refinement levels away may
        // share an edge without aligning, and no transition scheme covers that. Refuse the
        // offending vertex rather than extend the schemes, which is the remedy the paper chose.
        // Only weak balancing admits the jump at all, so the rule is skipped otherwise.
        let mut refused = HashSet::new();
        if matches!(self.balanced, Balancing::Weak(_)) {
            for &(_, corner, required) in coarse_nodes.iter() {
                if !required {
                    continue;
                }
                for bits in 0..1usize << D {
                    let vertex: [i32; D] =
                        from_fn(|axis| corner[axis] + ((bits >> axis) & 1) as i32);
                    if self.straddling_jump(&vertex, coarse) {
                        refused.insert(vertex);
                    }
                }
            }
        }
        let instance = Instance::new(
            coarse_nodes
                .iter()
                .map(|&(_, corner, required)| (corner, required))
                .collect(),
            refused,
        );
        let (assignment, _) = instance.solve();
        debug_assert!(instance.feasible(&assignment));
        self.pairing_vertices
            .extend(assignment.iter().map(|vertex| {
                let mut absolute = [0; D];
                for (axis, coordinate) in absolute.iter_mut().enumerate() {
                    *coordinate = vertex[axis] as usize * coarse;
                }
                (absolute, coarse)
            }));
        let mut paired = true;
        for (index, corner, required) in coarse_nodes {
            if required {
                continue;
            }
            let split = (0..1usize << D).any(|bits| {
                let mut vertex = corner;
                for (axis, coordinate) in vertex.iter_mut().enumerate() {
                    *coordinate += ((bits >> axis) & 1) as i32;
                }
                assignment.contains(&vertex)
            });
            if split {
                paired = false;
                self.subdivide(index)?;
            }
        }
        Ok(paired)
    }
    /// Whether the cluster this vertex would create straddles the doubled grid along some axis
    /// while the cells touching one of its edges there span two refinement levels.
    ///
    /// A cluster covers two cells, so along an axis it lines up with the doubled grid exactly
    /// when its vertex is odd there. Regular pairing's vertices always are, being node centres,
    /// which is why regular never meets this; generalized ones need not be. A four-fold span at
    /// the edge is in turn the jump only weak balancing admits. Neither alone is a problem, and
    /// the rule is inert unless both hold.
    fn straddling_jump(&self, vertex: &[i32; D], coarse: usize) -> bool {
        let root = &self[U::from(0)];
        let extent = root.length.into() as i64;
        let low: [i64; D] = from_fn(|axis| root.corner[axis].into() as i64);
        let length = coarse as i64;
        let center: [i64; D] = from_fn(|axis| vertex[axis] as i64 * length);
        let leaf_at = |point: [i64; D]| -> Option<U> {
            if (0..D).any(|axis| point[axis] < low[axis] || point[axis] >= low[axis] + extent) {
                return None;
            }
            let inside: [usize; D] = from_fn(|axis| point[axis] as usize);
            Some(U::from(leaf_containing(self, &inside)))
        };
        (0..D).filter(|&along| vertex[along] % 2 == 0).any(|along| {
            let others: Vec<usize> = (0..D).filter(|&axis| axis != along).collect();
            // Every grid line running this way that the cluster's own cells meet: its outer
            // edges, and the lines along the middle of each face, where the wedges anchored
            // on a coarse pair rather than on the cluster itself sit.
            (0..3usize.pow(others.len() as u32)).any(|line| {
                let sides: Vec<i64> = others
                    .iter()
                    .enumerate()
                    .map(|(slot, _)| (line / 3usize.pow(slot as u32)) % 3)
                    .map(|digit| digit as i64 - 1)
                    .collect();
                if sides.iter().all(|&side| side == 0) {
                    return false;
                }
                let halves: [Vec<(i64, U)>; 2] = from_fn(|half| {
                    let mut cells = Vec::new();
                    for shifts in 0..1usize << others.len() {
                        let mut point = center;
                        for (slot, &axis) in others.iter().enumerate() {
                            let shift = if (shifts >> slot) & 1 == 1 { 1 } else { -1 };
                            point[axis] += sides[slot] * length + shift;
                        }
                        point[along] += -length + half as i64 * length + length / 2;
                        if let Some(leaf) = leaf_at(point) {
                            cells.push((self[leaf].length.into() as i64, leaf))
                        }
                    }
                    cells.sort_unstable_by_key(|&(level, _)| level);
                    cells
                });
                // Only a configuration that changes partway along the line strands a wedge; one
                // that spans two levels uniformly is still a single case some template owns.
                let levels = |half: usize| -> Vec<i64> {
                    halves[half].iter().map(|&(level, _)| level).collect()
                };
                if levels(0) == levels(1) {
                    return false;
                }
                // The stranding case spans two refinement levels across the line - the paper's
                // "refinement n against n - 2", which only weak balancing admits. Splitting the
                // coarsest cell involved is what restores alignment.
                let cells = halves.concat();
                match cells.iter().map(|&(level, _)| level).min() {
                    Some(least) => cells.iter().any(|&(level, _)| level >= 4 * least),
                    None => false,
                }
            })
        })
    }
}
