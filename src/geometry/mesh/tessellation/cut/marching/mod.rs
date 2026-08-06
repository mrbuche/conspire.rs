#[cfg(test)]
mod test;

mod polyhedron;
mod project;
mod split;

use super::{DIRECTIONS, lattice::Lattice};
use crate::{
    geometry::{
        Coordinate, CoordinatesRef,
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{FxHashMap, Scalar, Tensor},
};

/// Where a boundary vertex sits on an edge whose ends straddle the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Placement {
    /// Halfway along, after Dhondt and Protais. The vertex is then as far
    /// from either end as it can be, so no cell can degenerate, at the cost
    /// of a boundary that converges only to first order.
    Midpoint,
    /// Where the surface actually crosses, after Tong and Zhang, held off
    /// either end by the given fraction of the edge so that the cell cannot
    /// degenerate. Converges to second order.
    Crossing(Scalar),
}

/// How the boundary is placed and then drawn onto the surface.
///
/// The default holds the crossings a fifth of an edge off either end and lets
/// the boundary settle for seven tenths of the quality it was cut with, which
/// on a bone leaves the mesh within a hundredth of a cell of the surface.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Marching {
    pub placement: Placement,
    /// The share of its scaled Jacobian every hexahedron must keep while the
    /// boundary is drawn onto the surface, or `None` to leave it as cut.
    pub keep: Option<Scalar>,
}

impl Default for Marching {
    fn default() -> Self {
        Self {
            placement: Placement::Crossing(0.2),
            keep: Some(0.7),
        }
    }
}

/// A lattice corner, identified by its index rather than its position so that
/// cells agree on it without comparing coordinates.
pub(super) type Corner = [usize; D];

/// A vertex of the polyhedron a cell is clipped to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(super) enum Vertex {
    Inside(Corner),
    Boundary([Corner; 2]),
}

impl Vertex {
    pub(super) fn boundary(one: Corner, two: Corner) -> Self {
        if one < two {
            Self::Boundary([one, two])
        } else {
            Self::Boundary([two, one])
        }
    }
}

pub(super) struct Signs {
    inside: FxHashMap<Corner, bool>,
    origin: Coordinate<D>,
    spacing: Scalar,
}

impl Signs {
    pub(super) fn at(&self, corner: Corner) -> bool {
        self.inside[&corner]
    }
    pub(super) fn point(&self, corner: Corner) -> Coordinate<D> {
        Coordinate::const_from(std::array::from_fn(|d| {
            self.origin[d] + corner[d] as Scalar * self.spacing
        }))
    }
}

impl Tessellation {
    /// Signs every corner of every occupied cell, strictly inside or strictly
    /// outside, which the cut relies on to never meet the surface at a corner.
    pub(super) fn signs(&self, lattice: &Lattice) -> Result<Signs, &'static str> {
        let surface = self.mesh();
        let coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: CoordinatesRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let (origin, spacing) = lattice.frame();
        let mut signs = Signs {
            inside: FxHashMap::default(),
            origin,
            spacing,
        };
        let mut corners: Vec<Corner> = lattice
            .cells()
            .iter()
            .flat_map(|&([i, j, k], _)| CORNERS.map(|[a, b, c]| [i + a, j + b, k + c]))
            .collect();
        corners.sort_unstable_by_key(|&[i, j, k]| (k, j, i));
        corners.dedup();
        let guard = super::CROSSING_TOLERANCE.max(spacing * 1.0e-6);
        for corner in corners {
            let point = signs.point(corner);
            let (closest, _) = bvh
                .closest_point(&point, coordinates, &elements)
                .ok_or("empty tessellation")?;
            if (&closest - &point).norm() < guard {
                return Err("a lattice corner lies on the surface");
            }
            let inside = self.encloses(&point, coordinates, &elements, &normals, &directions);
            signs.inside.insert(corner, inside);
        }
        Ok(signs)
    }
    /// Meshes this tessellation with hexahedra alone, by clipping every cell
    /// of a uniform lattice to the surface and splitting what is left about
    /// its midpoints.
    ///
    /// Passing a share to `draw` then moves the boundary onto the surface,
    /// as far as leaves every hexahedron holding that much of the scaled
    /// Jacobian it was cut with.
    pub fn marching_hex(
        &self,
        spacing: Scalar,
        marching: Marching,
    ) -> Result<Mesh<D>, &'static str> {
        let Marching { placement, keep } = marching;
        if let Placement::Crossing(guard) = placement
            && !(0.0..0.5).contains(&guard)
        {
            return Err("crossing guard must be within [0, 0.5)");
        }
        if let Some(keep) = keep
            && !(0.0..=1.0).contains(&keep)
        {
            return Err("the share of quality kept must be within [0, 1]");
        }
        // A corner sitting on the surface has no sign, and a lattice laid on
        // round numbers meets one readily, so it is moved until none does.
        let (lattice, signs) = SHIFTS
            .iter()
            .find_map(|&shift| {
                let lattice = match self.lattice_shifted(spacing, shift) {
                    Ok(lattice) => lattice,
                    Err(error) => return Some(Err(error)),
                };
                match self.signs(&lattice) {
                    Ok(signs) => Some(Ok((lattice, signs))),
                    Err("a lattice corner lies on the surface") => None,
                    Err(error) => Some(Err(error)),
                }
            })
            .unwrap_or(Err("every lattice tried meets the surface at a corner"))?;
        let cells = self.polyhedra(&lattice, &signs)?;
        if cells.is_empty() {
            return Err("no cell of the lattice has a corner inside the surface");
        }
        let points = self.placements(&cells, &signs, placement)?;
        split::hexahedra(cells, &points, keep.map(|keep| (self, keep)))
    }
}

/// Fractions of a cell to move the lattice by, in turn, until no corner of it
/// lies on the surface.
const SHIFTS: [[Scalar; D]; 5] = [
    [0.0, 0.0, 0.0],
    [0.013_717, 0.007_193, 0.002_971],
    [0.041_351, 0.023_887, 0.011_729],
    [0.097_153, 0.061_771, 0.033_413],
    [0.187_411, 0.140_412, 0.092_153],
];

pub(super) const CORNERS: [[usize; D]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [1, 1, 1],
    [0, 1, 1],
];

/// The six faces of a cell, each wound so that it is seen anticlockwise from
/// outside the cell.
pub(super) const FACES: [[usize; 4]; 6] = [
    [0, 3, 2, 1],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];
