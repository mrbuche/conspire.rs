#[cfg(test)]
mod test;

use super::{
    Fitting, Peeled,
    fit::{Facets, Oracle},
    merge,
    mixed::face_size,
};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, PrimitiveConnectivity, Tessellation, Verdict},
    },
    math::{Quantity, Scalar, Tensor, TensorVec},
    units::Length,
};
use std::array::from_fn;

/// Shell hexahedra whose worst scaled Jacobian falls below this, and whose
/// outer face meets a feature, are candidates for pyramid fans.
const BOWTIE: Scalar = 0.1;

/// Rings of neighbouring nodes, around each converted cell, freed in the refit.
const RINGS: usize = 2;

/// The shortest distance from `point` to the polyline through `curve`.
pub(crate) fn distance_to_curve(point: &Coordinate<3>, curve: &[Coordinate<3>]) -> Scalar {
    let p: [Scalar; 3] = from_fn(|k| point[k].value());
    curve
        .windows(2)
        .map(|pair| {
            let a: [Scalar; 3] = from_fn(|k| pair[0][k].value());
            let b: [Scalar; 3] = from_fn(|k| pair[1][k].value());
            let ab: [Scalar; 3] = from_fn(|k| b[k] - a[k]);
            let length: Scalar = ab.iter().map(|x| x * x).sum();
            let along = if length > 0.0 {
                ((0..3).map(|k| (p[k] - a[k]) * ab[k]).sum::<Scalar>() / length).clamp(0.0, 1.0)
            } else {
                0.0
            };
            (0..3)
                .map(|k| (p[k] - a[k] - along * ab[k]).powi(2))
                .sum::<Scalar>()
                .sqrt()
        })
        .fold(Scalar::INFINITY, Scalar::min)
}

fn worst(mesh: &Mesh<3>) -> Scalar {
    mesh.minimum_scaled_jacobians()
        .iter()
        .flatten()
        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality))
}

impl Mesh<3> {
    fn snap<O: Oracle>(&mut self, oracle: &O, layer: &[usize]) -> Result<(), &'static str> {
        let coordinates = self.coordinates.members_mut();
        layer.iter().try_for_each(|&node| {
            let (point, _) = oracle
                .project(&coordinates[node])
                .ok_or("no projection onto target surface")?;
            coordinates[node] = point;
            Ok(())
        })
    }
    /// Adds a buffer layer as [`buffer`](Self::buffer) does, then, for
    /// [`Fitting::Snap`], replaces the shell hexahedra that projection left
    /// badly shaped along a feature with pyramid fans.
    ///
    /// Every candidate is converted and refitted, and a fan is kept only if it
    /// beats the hexahedron it replaces. The result is never worse, by
    /// minimum scaled Jacobian, than the all-hexahedral buffer.
    pub fn buffer_targeted(
        self,
        target: &Tessellation,
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        self.targeted(target, fitting, BOWTIE)
    }
    fn targeted(
        self,
        target: &Tessellation,
        fitting: Fitting,
        threshold: Scalar,
    ) -> Result<Self, &'static str> {
        self.targeted_with(
            &Facets::new(target),
            &[],
            |size| {
                let index = target.features().index(size);
                move |point: &Coordinate<3>, size| {
                    index.nearest_corner(point, size).is_some()
                        || index.nearest_crease(point, size).is_some()
                }
            },
            fitting,
            threshold,
        )
    }

    /// [`targeted_with`](Self::targeted_with), where a feature is any of the
    /// crease `curves`.
    pub(crate) fn targeted_along<O: Oracle>(
        self,
        oracle: &O,
        creases: &[(Vec<Coordinate<3>>, Vec<usize>)],
        curves: &[(Vec<Coordinate<3>>, Vec<usize>)],
        fitting: Fitting,
    ) -> Result<Self, &'static str> {
        self.targeted_with(
            oracle,
            creases,
            |_| {
                |point: &Coordinate<3>, size: Quantity<Length>| {
                    curves
                        .iter()
                        .any(|(curve, _)| distance_to_curve(point, curve) < size.value())
                }
            },
            fitting,
            BOWTIE,
        )
    }

    /// [`buffer_targeted`](Self::buffer_targeted) against any fit [`Oracle`].
    ///
    /// `creases` constrain the fit as in [`buffer_with`](Self::buffer_with).
    /// `near` is built from the largest boundary face size and says whether
    /// a point is within a face size of a feature.
    pub(crate) fn targeted_with<O, G, N>(
        mut self,
        oracle: &O,
        creases: &[(Vec<Coordinate<3>>, Vec<usize>)],
        near: G,
        fitting: Fitting,
        threshold: Scalar,
    ) -> Result<Self, &'static str>
    where
        O: Oracle,
        G: FnOnce(Quantity<Length>) -> N,
        N: Fn(&Coordinate<3>, Quantity<Length>) -> bool,
    {
        self.restrict()?;
        let boundary = self.exterior_faces();
        let Peeled {
            mut connectivities,
            coordinates,
            count,
            duplicates,
            layer,
        } = self.peel(&boundary, 4, "non-quadrilateral boundary face")?;
        let sizes: Vec<Quantity<Length>> = boundary
            .iter()
            .map(|face| face_size(face, &coordinates))
            .collect();
        let near = near(
            sizes
                .iter()
                .copied()
                .fold(Quantity::new(0.0), Quantity::max),
        );
        let shell: Vec<[usize; 8]> = boundary
            .iter()
            .map(|face| {
                let n: [usize; 4] = from_fn(|i| face[i]);
                let m: [usize; 4] = n.map(|node| duplicates[&node]);
                [n[0], n[1], n[2], n[3], m[0], m[1], m[2], m[3]]
            })
            .collect();
        merge(
            &mut connectivities,
            shell.clone(),
            |connectivity| matches!(connectivity, Connectivity::Hexahedral(_)),
            Connectivity::Hexahedral,
        )?;
        let mut mesh = Self::from((connectivities, coordinates));
        let nodes: Vec<usize> = layer.iter().copied().chain(0..count).collect();
        mesh.fit(&nodes, oracle, creases)?;
        let Fitting::Snap = fitting else {
            return Ok(mesh);
        };
        mesh.snap(oracle, &layer)?;
        mesh.fit(&(0..count).collect::<Vec<_>>(), oracle, creases)?;
        if !mesh
            .connectivities()
            .iter()
            .all(|connectivity| matches!(connectivity, Connectivity::Hexahedral(_)))
        {
            return Ok(mesh);
        }
        let qualities = &mesh.minimum_scaled_jacobians()[mesh.connectivities().len() - 1];
        let first = qualities.len() - shell.len();
        let old: Vec<Scalar> = qualities[first..].to_vec();
        let bad: Vec<bool> = shell
            .iter()
            .enumerate()
            .map(|(face, cell)| {
                let centroid = cell[4..]
                    .iter()
                    .map(|&node| &mesh.coordinates()[node])
                    .sum::<Coordinate<3>>()
                    / 4.0;
                old[face] < threshold && near(&centroid, sizes[face])
            })
            .collect();
        if !bad.iter().any(|&flag| flag) {
            return Ok(mesh);
        }
        let baseline = worst(&mesh);
        let (fitted, fitted_coordinates) = mesh.into();
        let blocks: Vec<Vec<[usize; 8]>> = fitted
            .into_members()
            .into_iter()
            .map(|connectivity| {
                PrimitiveConnectivity::<3, 8>::try_from(connectivity)
                    .map(|block| block.into_iter().collect())
            })
            .collect::<Result<_, _>>()?;
        let hexahedra = |blocks: Vec<Vec<[usize; 8]>>| -> Vec<Connectivity> {
            blocks
                .into_iter()
                .map(|block| Connectivity::Hexahedral(block.into()))
                .collect()
        };
        let assemble = |flags: &[bool]| -> Result<Self, &'static str> {
            let mut coordinates = fitted_coordinates.clone();
            let fitted_count = coordinates.len();
            let mut layer = layer.clone();
            let mut blocks = blocks.clone();
            let mut pyramids: Vec<[usize; 5]> = Vec::new();
            let last = blocks.last_mut().ok_or("no hexahedral block")?;
            last.truncate(first);
            shell.iter().zip(flags).for_each(|(cell, &flag)| {
                if flag {
                    let apex = coordinates.len();
                    let centroid = cell[4..]
                        .iter()
                        .map(|&node| &coordinates[node])
                        .sum::<Coordinate<3>>()
                        / 4.0;
                    coordinates.push(centroid);
                    layer.push(apex);
                    let (n, m) = (&cell[..4], &cell[4..]);
                    pyramids.push([n[0], n[1], n[2], n[3], apex]);
                    (0..4).for_each(|i| {
                        let j = (i + 1) % 4;
                        pyramids.push([n[j], n[i], m[i], m[j], apex]);
                    });
                } else {
                    last.push(*cell);
                }
            });
            let mut connectivities = hexahedra(blocks);
            connectivities.push(Connectivity::Pyramidal(pyramids.into()));
            let mut mesh = Self::from((connectivities, coordinates));
            let neighbors = mesh.node_node_connectivity().to_vec();
            let mut free = vec![false; mesh.number_of_nodes()];
            let mut front: Vec<usize> = (fitted_count..free.len())
                .chain(
                    shell
                        .iter()
                        .zip(flags)
                        .filter(|&(_, &flag)| flag)
                        .flat_map(|(cell, _)| cell.iter().copied()),
                )
                .collect();
            front.iter().for_each(|&node| free[node] = true);
            for _ in 0..RINGS {
                front = front
                    .into_iter()
                    .flat_map(|node| neighbors[node].iter().copied())
                    .filter(|&node| !std::mem::replace(&mut free[node], true))
                    .collect();
            }
            let free: Vec<usize> = (0..free.len()).filter(|&node| free[node]).collect();
            let layer: Vec<usize> = layer
                .into_iter()
                .filter(|node| free.binary_search(node).is_ok())
                .collect();
            mesh.fit(&free, oracle, creases)?;
            mesh.snap(oracle, &layer)?;
            let core: Vec<usize> = free.into_iter().filter(|&node| node < count).collect();
            mesh.fit(&core, oracle, creases)?;
            Ok(mesh)
        };
        let unchanged = || Self::from((hexahedra(blocks.clone()), fitted_coordinates.clone()));
        let trial = assemble(&bad)?;
        let fans = trial
            .minimum_scaled_jacobians()
            .pop()
            .ok_or("no pyramidal block")?;
        let mut fan = 0;
        let accepted: Vec<bool> = bad
            .iter()
            .enumerate()
            .map(|(face, &flag)| {
                flag && {
                    let quality = fans[5 * fan..5 * fan + 5]
                        .iter()
                        .fold(Scalar::INFINITY, |worst, &quality| worst.min(quality));
                    fan += 1;
                    quality >= old[face]
                }
            })
            .collect();
        let candidate = if accepted == bad {
            trial
        } else if accepted.iter().any(|&flag| flag) {
            assemble(&accepted)?
        } else {
            return Ok(unchanged());
        };
        Ok(if worst(&candidate) >= baseline {
            candidate
        } else {
            unchanged()
        })
    }
}
