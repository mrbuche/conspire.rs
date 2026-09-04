#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::{Connectivity, Mesh, Verdict},
    math::{Scalar, Tensor, TensorVec},
};
use std::collections::{BTreeSet, HashMap};

/// The six faces of a hexahedron, outward-oriented, as quads of local nodes.
const FACES: [[usize; 4]; 6] = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 3, 2, 1],
    [4, 5, 6, 7],
];

impl Mesh<3> {
    /// Wraps the sub-region given by `region` (element indices into the single
    /// hexahedral block) in a fresh layer of hexahedra, so its nodes gain an
    /// interior twin the fit can move independently — a local pillow, after
    /// Protais et al. §4.1.2.
    ///
    /// Returns the new twin nodes (the layer's inner ring). The region's
    /// boundary must be a closed 2-manifold of quadrilaterals.
    pub(crate) fn pillow(&mut self, region: &[usize]) -> Result<Vec<usize>, &'static str> {
        let mut hexes: Vec<[usize; 8]> = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(block) => hexes.extend(block.iter().copied()),
                _ => return Err("pillow requires a hexahedral mesh"),
            }
        }
        let mut inside = vec![false; hexes.len()];
        for &cell in region {
            *inside.get_mut(cell).ok_or("region cell out of range")? = true;
        }
        // Oriented boundary faces: a region hex's face not shared with another
        // region hex.
        let mut seen: HashMap<[usize; 4], usize> = HashMap::new();
        for (cell, hex) in hexes.iter().enumerate() {
            if !inside[cell] {
                continue;
            }
            for face in FACES {
                let mut key = face.map(|local| hex[local]);
                key.sort_unstable();
                *seen.entry(key).or_insert(0) += 1;
            }
        }
        let mut boundary: Vec<[usize; 4]> = Vec::new();
        for (cell, hex) in hexes.iter().enumerate() {
            if !inside[cell] {
                continue;
            }
            for face in FACES {
                let oriented = face.map(|local| hex[local]);
                let mut key = oriented;
                key.sort_unstable();
                if seen[&key] == 1 {
                    boundary.push(oriented);
                }
            }
        }
        if boundary.is_empty() {
            return Err("region has no boundary");
        }
        let mut edges: HashMap<[usize; 2], usize> = HashMap::new();
        for face in &boundary {
            for i in 0..4 {
                let mut edge = [face[i], face[(i + 1) % 4]];
                edge.sort_unstable();
                *edges.entry(edge).or_insert(0) += 1;
            }
        }
        if edges.values().any(|&count| count != 2) {
            return Err("region boundary is not a closed manifold");
        }
        // Twin every boundary node.
        let mut coordinates = self.coordinates().clone();
        let mut twin: HashMap<usize, usize> = HashMap::new();
        let mut twins: Vec<usize> = Vec::new();
        for &node in boundary.iter().flatten() {
            twin.entry(node).or_insert_with(|| {
                let new = coordinates.len();
                coordinates.push(coordinates[node].clone());
                twins.push(new);
                new
            });
        }
        // Rewire the region onto its twins; the rest of the mesh keeps the
        // originals, and a sheet hex spans the two along each boundary face.
        for (cell, hex) in hexes.iter_mut().enumerate() {
            if inside[cell] {
                for node in hex.iter_mut() {
                    if let Some(&new) = twin.get(node) {
                        *node = new;
                    }
                }
            }
        }
        for face in boundary {
            let [a, b, c, d] = face;
            hexes.push([a, b, c, d, twin[&a], twin[&b], twin[&c], twin[&d]]);
        }
        *self = Mesh::from((vec![Connectivity::Hexahedral(hexes.into())], coordinates));
        Ok(twins)
    }
    /// Pillows the hexahedra around the boundary nodes where an exterior quad
    /// opens past `alpha` radians — the Protais et al. §4.1.2 signature of one
    /// hex forced to place two edges along the same feature — restricted to
    /// cells whose scaled Jacobian is below `gate`.
    ///
    /// Each face-connected component of that set is pillowed on its own, so a
    /// worst spot gets relief even where the full flagged region would not
    /// bound a manifold. Best-effort: a component that still pinches is left as
    /// fitted, and a relief request never blocks the mesh. Returns the twin
    /// nodes to free in a re-fit.
    pub(crate) fn relieve_open_angles(&mut self, alpha: Scalar, gate: Scalar) -> Vec<usize> {
        let flagged = open_angle_nodes(self, alpha);
        if flagged.is_empty() {
            return Vec::new();
        }
        let mut hexes: Vec<[usize; 8]> = Vec::new();
        for block in self.iter() {
            match block {
                Connectivity::Hexahedral(block) => hexes.extend(block.iter().copied()),
                _ => return Vec::new(),
            }
        }
        let scaled: Vec<Scalar> = self
            .minimum_scaled_jacobians()
            .into_iter()
            .flatten()
            .collect();
        let region: Vec<usize> = hexes
            .iter()
            .enumerate()
            .filter(|(cell, hex)| {
                scaled[*cell] < gate && hex.iter().any(|node| flagged.contains(node))
            })
            .map(|(cell, _)| cell)
            .collect();
        if region.is_empty() {
            return Vec::new();
        }
        let mut twins = Vec::new();
        for component in components(&region, &hexes) {
            twins.extend(self.pillow(&component).unwrap_or_default());
        }
        twins
    }
}

/// Face-connected components of a set of hexahedra.
fn components(region: &[usize], hexes: &[[usize; 8]]) -> Vec<Vec<usize>> {
    let mut parent: Vec<usize> = (0..region.len()).collect();
    fn root(parent: &mut [usize], mut node: usize) -> usize {
        while parent[node] != node {
            parent[node] = parent[parent[node]];
            node = parent[node];
        }
        node
    }
    let mut owners: HashMap<[usize; 4], usize> = HashMap::new();
    for (index, &cell) in region.iter().enumerate() {
        for face in FACES {
            let mut key = face.map(|local| hexes[cell][local]);
            key.sort_unstable();
            match owners.get(&key) {
                Some(&other) => {
                    let (a, b) = (root(&mut parent, index), root(&mut parent, other));
                    parent[a] = b;
                }
                None => {
                    owners.insert(key, index);
                }
            }
        }
    }
    let mut grouped: HashMap<usize, Vec<usize>> = HashMap::new();
    for (index, &cell) in region.iter().enumerate() {
        let r = root(&mut parent, index);
        grouped.entry(r).or_default().push(cell);
    }
    grouped.into_values().collect()
}

/// Boundary nodes at which an exterior quad opens past `alpha` radians.
pub(crate) fn open_angle_nodes(mesh: &Mesh<3>, alpha: Scalar) -> BTreeSet<usize> {
    let coordinates = mesh.coordinates();
    let mut flagged = BTreeSet::new();
    for face in mesh.exterior_faces() {
        let count = face.len();
        for corner in 0..count {
            let here = &coordinates[face[corner]];
            let before = &coordinates[face[(corner + count - 1) % count]];
            let after = &coordinates[face[(corner + 1) % count]];
            let one = (before - here).normalized();
            let two = (after - here).normalized();
            if (&one * &two).value().clamp(-1.0, 1.0).acos() > alpha {
                flagged.insert(face[corner]);
            }
        }
    }
    flagged
}
