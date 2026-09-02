#[cfg(test)]
mod test;

mod fit;
mod restrict;

use super::{Connectivity, Mesh, PrimitiveConnectivity, Tessellation};
use crate::{
    geometry::Coordinates,
    math::{Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, hash_map::Entry},
};

/// The four faces of a tetrahedron, as triples of local node indices.
const TET_FACES: [[usize; 3]; 4] = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]];

/// A mesh peeled open along its boundary, with that boundary's nodes
/// duplicated so a layer of elements can span the two copies.
struct Peeled {
    connectivities: Vec<Connectivity>,
    coordinates: Coordinates<3>,
    count: usize,
    duplicates: HashMap<usize, usize>,
    layer: Vec<usize>,
}

/// Splits the prism standing on an outward boundary triangle into three
/// tetrahedra.
///
/// Each lateral quadrilateral is cut by the diagonal running from its
/// lower-numbered base node to the duplicate of the higher one. That depends
/// on the edge's two nodes alone, so the prisms either side of a boundary edge
/// cut their shared quadrilateral the same way, and since node numbers are a
/// total order the three diagonals can never wind around the prism, which is
/// what would leave it untetrahedralizable without a new node.
fn prism(face: &[usize], duplicates: &HashMap<usize, usize>) -> [[usize; 4]; 3] {
    let first = (0..3)
        .min_by_key(|&i| face[i])
        .expect("empty boundary face");
    let [p0, p1, p2]: [usize; 3] = from_fn(|i| face[(first + i) % 3]);
    let [q0, q1, q2] = [p0, p1, p2].map(|node| duplicates[&node]);
    if p1 < p2 {
        [[p0, p1, p2, q2], [p0, p1, q2, q1], [p0, q1, q2, q0]]
    } else {
        [[p0, p1, p2, q1], [p0, p2, q2, q1], [p0, q2, q0, q1]]
    }
}

/// Adds `cells` to the last block of their own kind, or as a new one.
fn merge<const N: usize>(
    connectivities: &mut Vec<Connectivity>,
    cells: Vec<[usize; N]>,
    of_kind: fn(&Connectivity) -> bool,
    variant: fn(PrimitiveConnectivity<3, N>) -> Connectivity,
) -> Result<(), &'static str>
where
    PrimitiveConnectivity<3, N>: TryFrom<Connectivity, Error = &'static str>,
{
    match connectivities.iter().rposition(of_kind) {
        Some(index) => {
            let block = PrimitiveConnectivity::<3, N>::try_from(connectivities.remove(index))?;
            connectivities.insert(
                index,
                variant(block.into_iter().chain(cells).collect::<Vec<_>>().into()),
            )
        }
        None => connectivities.push(variant(cells.into())),
    }
    Ok(())
}

/// Drops tetrahedra until the mesh boundary is edge-manifold: every boundary
/// edge carried by exactly two boundary faces.
///
/// [`trim`](Tessellation::trim) keeps or discards a background cell by the
/// signed distances at its nodes alone, with no topological guard, so a
/// tetrahedral background can be left pinched along an edge or hanging by one.
/// Such a boundary cannot be [peeled](Mesh::peel), so each tetrahedron that
/// carries a boundary face on a non-manifold edge is removed, and the check
/// repeated, until the boundary closes up. A hexahedral background trimmed the
/// same way rarely pinches, and [`buffer`](Mesh::buffer) leaves it alone.
fn manifold_boundary(mut mesh: Mesh<3>) -> Result<Mesh<3>, &'static str> {
    for _ in 0..64 {
        let tets: Vec<[usize; 4]> = mesh
            .iter()
            .flatten()
            .map(|tet| from_fn(|i| tet[i]))
            .collect();
        let mut face_tets: HashMap<[usize; 3], Vec<usize>> = HashMap::new();
        for (element, tet) in tets.iter().enumerate() {
            for face in TET_FACES {
                let mut key = face.map(|node| tet[node]);
                key.sort_unstable();
                face_tets.entry(key).or_default().push(element);
            }
        }
        let mut edge_faces: HashMap<[usize; 2], u32> = HashMap::new();
        for (face, owners) in &face_tets {
            if owners.len() == 1 {
                for [a, b] in [[0, 1], [1, 2], [2, 0]] {
                    let mut edge = [face[a], face[b]];
                    edge.sort_unstable();
                    *edge_faces.entry(edge).or_insert(0) += 1;
                }
            }
        }
        let bad: HashSet<[usize; 2]> = edge_faces
            .into_iter()
            .filter_map(|(edge, count)| (count != 2).then_some(edge))
            .collect();
        if bad.is_empty() {
            return Ok(mesh);
        }
        let mut discard: HashSet<usize> = HashSet::new();
        for (face, owners) in &face_tets {
            if owners.len() == 1
                && [[0, 1], [1, 2], [2, 0]].iter().any(|&[a, b]| {
                    let mut edge = [face[a], face[b]];
                    edge.sort_unstable();
                    bad.contains(&edge)
                })
            {
                discard.insert(owners[0]);
            }
        }
        if discard.is_empty() {
            return Err("non-manifold boundary");
        }
        let mut element = 0;
        mesh.retain_elements(|_, _, _| {
            let keep = !discard.contains(&element);
            element += 1;
            keep
        });
    }
    Err("non-manifold boundary")
}

/// Constraint on how the buffer layer approaches the target surface.
#[derive(Clone, Copy, Debug)]
pub enum Fitting {
    /// The layer settles wherever the quality and fit energies balance.
    Soft,
    /// The layer settles as above, but is then projected onto the surface,
    /// after which the interior relaxes.
    Snap,
}

impl Mesh<3> {
    pub fn buffer(mut self, target: &Tessellation, fitting: Fitting) -> Result<Self, &'static str> {
        self.restrict()?;
        let boundary = self.exterior_faces();
        let Peeled {
            mut connectivities,
            coordinates,
            count,
            duplicates,
            layer,
        } = self.peel(&boundary, 4, "non-quadrilateral boundary face")?;
        let cells = boundary
            .iter()
            .map(|face| {
                [
                    face[0],
                    face[1],
                    face[2],
                    face[3],
                    duplicates[&face[0]],
                    duplicates[&face[1]],
                    duplicates[&face[2]],
                    duplicates[&face[3]],
                ]
            })
            .collect::<Vec<_>>();
        merge(
            &mut connectivities,
            cells,
            |connectivity| matches!(connectivity, Connectivity::Hexahedral(_)),
            Connectivity::Hexahedral,
        )?;
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit(&nodes, target)?;
        if let Fitting::Snap = fitting {
            mesh.project(target, &layer)?;
            mesh.fit(&(0..count).collect::<Vec<_>>(), target)?;
        }
        Ok(mesh)
    }
    /// Adds a buffer layer of tetrahedra to a tetrahedral mesh and fits it to
    /// the target.
    ///
    /// The counterpart of [`buffer`](Self::buffer), differing in that each
    /// boundary triangle raises a prism split into three tetrahedra rather
    /// than one hexahedron, so the result stays a single tetrahedral block.
    ///
    /// It also runs no clearance pre-pass. [`restrict`](Self::restrict) is
    /// defined on hexahedral boundary quadrilaterals and has no tetrahedral
    /// analogue yet, so a boundary leaving some node no feasible direction is
    /// fitted here rather than pruned first.
    pub fn buffer_tets(
        self,
        target: &Tessellation,
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        let cleaned = manifold_boundary(self)?;
        let boundary = cleaned.exterior_faces();
        let Peeled {
            mut connectivities,
            coordinates,
            count,
            duplicates,
            layer,
        } = cleaned.peel(&boundary, 3, "non-triangular boundary face")?;
        let cells: Vec<[usize; 4]> = boundary
            .iter()
            .flat_map(|face| prism(face, &duplicates))
            .collect();
        merge(
            &mut connectivities,
            cells,
            |connectivity| matches!(connectivity, Connectivity::Tetrahedral(_)),
            Connectivity::Tetrahedral,
        )?;
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit_tets(&nodes, target)?;
        if let Fitting::Snap = fitting {
            mesh.project(target, &layer)?;
            mesh.fit_tets(&(0..count).collect::<Vec<_>>(), target)?;
        }
        Ok(mesh)
    }
    /// Checks the boundary is a manifold of `arity`-node faces and duplicates
    /// its nodes.
    fn peel(
        self,
        boundary: &[Vec<usize>],
        arity: usize,
        misshapen: &'static str,
    ) -> Result<Peeled, &'static str> {
        let mut edges = HashMap::new();
        boundary.iter().try_for_each(|face| {
            if face.len() != arity {
                return Err(misshapen);
            }
            (0..arity).for_each(|i| {
                let mut edge = [face[i], face[(i + 1) % arity]];
                edge.sort_unstable();
                *edges.entry(edge).or_insert(0u8) += 1;
            });
            Ok(())
        })?;
        if edges.values().any(|&count| count != 2) {
            return Err("non-manifold boundary");
        }
        let (connectivities, mut coordinates) = self.into();
        let connectivities = connectivities.into_members();
        let count = coordinates.len();
        let mut duplicates = HashMap::new();
        let mut layer = Vec::new();
        boundary.iter().flatten().for_each(|&node| {
            if let Entry::Vacant(slot) = duplicates.entry(node) {
                slot.insert(coordinates.len());
                layer.push(coordinates.len());
                let point = coordinates[node].clone();
                coordinates.push(point);
            }
        });
        Ok(Peeled {
            connectivities,
            coordinates,
            count,
            duplicates,
            layer,
        })
    }
    /// Moves the layer's nodes onto the closest point of the target.
    fn project(&mut self, target: &Tessellation, layer: &[usize]) -> Result<(), &'static str> {
        let surface = target.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = target.bvh();
        let coordinates = self.coordinates.members_mut();
        layer.iter().try_for_each(|&node| {
            let (point, _) = bvh
                .closest_point(&coordinates[node], surface_coordinates, &elements)
                .ok_or("empty tessellation")?;
            coordinates[node] = point;
            Ok(())
        })
    }
}
