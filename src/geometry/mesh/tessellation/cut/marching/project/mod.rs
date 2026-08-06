use crate::{
    geometry::{
        Coordinates,
        mesh::{
            quality::metrics::hexahedron::{bernstein, minimum_scaled_jacobian},
            tessellation::{D, Tessellation},
        },
    },
    math::{FxHashMap, Scalar, Tensor},
};

const BISECTIONS: usize = 8;
const KEEP: Scalar = 0.5;
const PASSES: usize = 4;

const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

/// The nodes on faces belonging to only one hexahedron, which for a mesh cut
/// from a lattice are exactly the ones approximating the surface.
fn boundary(hexes: &[[usize; 8]]) -> Vec<usize> {
    let mut counts = FxHashMap::default();
    hexes.iter().for_each(|hex| {
        FACES.iter().for_each(|face| {
            let mut key = face.map(|local| hex[local]);
            key.sort_unstable();
            *counts.entry(key).or_insert(0u8) += 1
        })
    });
    let mut nodes: Vec<usize> = counts
        .iter()
        .filter(|&(_, &count)| count == 1)
        .flat_map(|(face, _)| *face)
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

impl Tessellation {
    /// Draws the boundary onto the surface, moving each node as far along the
    /// way as leaves every hexahedron it belongs to still certified.
    ///
    /// Each node is answered for on its own, against a check that proves
    /// rather than samples, so this settles no energy and needs no
    /// convergence: a node either moves or it does not, and the mesh is valid
    /// throughout either way.
    pub(super) fn draw_onto(&self, hexes: &[[usize; 8]], coordinates: &mut Coordinates<D>) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let nodes = boundary(hexes);
        let mut belongs: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
        hexes.iter().enumerate().for_each(|(index, hex)| {
            hex.iter().for_each(|&node| {
                belongs.entry(node).or_default().push(index);
            })
        });
        // Measured once, against the mesh as it was cut, so that quality
        // cannot ratchet down a share at a time as neighbouring nodes move.
        // The scaled Jacobian is what bounds quality, since a flattened cell
        // holds its Bernstein margin while losing its shape entirely; the
        // certificate is what still rules out an inversion anywhere within.
        let floors: Vec<Scalar> = hexes
            .iter()
            .map(|hex| KEEP * minimum_scaled_jacobian(hex, coordinates))
            .collect();
        for _ in 0..PASSES {
            for &node in nodes.iter() {
                let here = coordinates[node].clone();
                let Some((onto, _)) = bvh.closest_point(&here, surface_coordinates, &elements)
                else {
                    continue;
                };
                let along = &onto - &here;
                if along.norm() == 0.0 {
                    continue;
                }
                // Certifying alone would let a node travel until its cells
                // were on the point of degenerating, so each must keep a share
                // of the margin it was cut with.
                let certified = |coordinates: &Coordinates<D>| {
                    belongs[&node].iter().all(|&index| {
                        minimum_scaled_jacobian(&hexes[index], coordinates) >= floors[index]
                            && bernstein::certifies(&hexes[index], coordinates)
                    })
                };
                let mut low = 0.0;
                let mut high = 1.0;
                coordinates[node] = onto.clone();
                if !certified(coordinates) {
                    for _ in 0..BISECTIONS {
                        let middle = (low + high) / 2.0;
                        coordinates[node] = &here + &(&along * middle);
                        if certified(coordinates) {
                            low = middle
                        } else {
                            high = middle
                        }
                    }
                    coordinates[node] = &here + &(&along * low);
                }
            }
        }
    }
    /// How far the boundary sits from the surface, at worst and on average,
    /// as a fraction of the cell size.
    #[cfg(test)]
    pub(super) fn conformance(
        &self,
        hexes: &[[usize; 8]],
        coordinates: &Coordinates<D>,
        spacing: Scalar,
    ) -> (Scalar, Scalar) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let nodes = boundary(hexes);
        let distances: Vec<Scalar> = nodes
            .iter()
            .filter_map(|&node| {
                bvh.closest_point(&coordinates[node], surface_coordinates, &elements)
                    .map(|(onto, _)| (&onto - &coordinates[node]).norm())
            })
            .collect();
        (
            distances.iter().cloned().fold(0.0, Scalar::max) / spacing,
            distances.iter().sum::<Scalar>() / distances.len() as Scalar / spacing,
        )
    }
}
