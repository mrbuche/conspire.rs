#[cfg(test)]
mod test;

mod polyhedron;
mod project;
mod split;

use super::{DIRECTIONS, lattice::Lattice};
use crate::{
    geometry::{
        Coordinate, DirectionsRef,
        grid::{Gradient, MarchingCubes, Voxels},
        mesh::{
            Fitting, Freedom, Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{FxHashMap, Quantity, Scalar, Tensor},
    units::Length,
};
use std::array::from_fn;

pub(super) use crate::geometry::grid::marching_cubes::separated::{
    CORNERS, Corner, Vertex, inside,
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

/// What to do with the boundary once the lattice has been cut and split.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Finish {
    /// Leave the boundary where the cut placed it.
    Cut,
    /// Draw each boundary node onto the surface by bisection, keeping the
    /// given share of the scaled Jacobian its incident hexahedra were cut
    /// with. Moves the boundary alone, no volume degrees of freedom.
    Draw(Scalar),
    /// Deform the mesh onto the surface by energy fitting, after Protais:
    /// vertices move through the volume, connectivity does not, and no
    /// elements are added. With [`Fitting::Snap`] the boundary is then
    /// projected onto the surface and the interior relaxes around it.
    Fit(Freedom, Fitting),
}

/// How the boundary is placed and then settled onto the surface.
///
/// The default holds the crossings a fifth of an edge off either end and
/// draws the boundary on for seven tenths of the quality it was cut with.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Marching {
    pub placement: Placement,
    pub finish: Finish,
}

impl Default for Marching {
    fn default() -> Self {
        Self {
            placement: Placement::Crossing(0.2),
            finish: Finish::Draw(0.7),
        }
    }
}

struct Field<'a> {
    volume: &'a Voxels<f64>,
    level: Scalar,
    gradient: Gradient,
}

impl Field<'_> {
    fn value(&self, corner: Corner) -> Scalar {
        self.volume.data()[self.volume.flat(corner)]
    }
}

pub(super) struct Signs<'a> {
    inside: FxHashMap<Corner, bool>,
    field: Option<Field<'a>>,
    origin: Coordinate<D>,
    spacing: [Quantity<Length>; D],
}

impl Signs<'_> {
    pub(super) fn at(&self, corner: Corner) -> bool {
        match &self.field {
            Some(field) => inside(field.value(corner), field.level, field.gradient),
            None => self.inside[&corner],
        }
    }
    pub(super) fn point(&self, corner: Corner) -> Coordinate<D> {
        Coordinate::from(from_fn(|d| {
            self.origin[d] + corner[d] as Scalar * self.spacing[d]
        }))
    }
    pub(super) fn fraction(&self, one: Corner, two: Corner) -> Option<Scalar> {
        self.field.as_ref().map(|field| {
            let (a, b) = (field.value(one), field.value(two));
            (field.level - a) / (b - a)
        })
    }
}

impl MarchingCubes {
    /// Meshes the object where a scalar field, sampled on a regular grid,
    /// reaches its level, with hexahedra alone, by clipping every cell of
    /// the grid and splitting what is left about its midpoints.
    ///
    /// The samples are the nodes of the lattice and the spacing may differ
    /// along each axis. The surface lies where the field is interpolated to
    /// cross the level, held off either end as `placement` says, and an
    /// object reaching the edge of the grid is cut flat there. The `method`,
    /// `step` and `degenerate` of the extraction do not apply.
    pub fn hexahedra(
        &self,
        volume: &Voxels<f64>,
        placement: Placement,
    ) -> Result<Mesh<D>, &'static str> {
        let nel = *volume.nel();
        if nel.iter().any(|&n| n < 2) {
            return Err("Input array must be at least 2x2x2.");
        }
        if self.step != 1 {
            return Err("Hexahedra take every sample.");
        }
        if let Placement::Crossing(guard) = placement
            && !(0.0..0.5).contains(&guard)
        {
            return Err("crossing guard must be within [0, 0.5)");
        }
        let level = self.resolve(volume)?;
        let signs = Signs {
            inside: FxHashMap::default(),
            field: Some(Field {
                volume,
                level,
                gradient: self.gradient,
            }),
            origin: Coordinate::from([Length::meters(0.0); D]),
            spacing: from_fn(|axis| self.spacing[axis]),
        };
        let cells = (0..nel[2] - 1).flat_map(|k| {
            (0..nel[1] - 1).flat_map(move |j| (0..nel[0] - 1).map(move |i| [i, j, k]))
        });
        let cells = polyhedron::polyhedra(cells, &signs)?;
        if cells.is_empty() {
            return Err("No cell of the grid has a sample within the object.");
        }
        let points = split::placements(None, &cells, &signs, placement)?;
        split::hexahedra(cells, &points, None)
    }
}

impl Tessellation {
    pub(super) fn signs(&self, lattice: &Lattice) -> Result<Signs<'_>, &'static str> {
        let surface = self.mesh();
        let coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: DirectionsRef<'_, D> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let (origin, spacing) = lattice.frame();
        let mut signs = Signs {
            inside: FxHashMap::default(),
            field: None,
            origin,
            spacing: [spacing; D],
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
        spacing: Quantity<Length>,
        marching: Marching,
    ) -> Result<Mesh<D>, &'static str> {
        let Marching { placement, finish } = marching;
        if let Placement::Crossing(guard) = placement
            && !(0.0..0.5).contains(&guard)
        {
            return Err("crossing guard must be within [0, 0.5)");
        }
        if let Finish::Draw(keep) = finish
            && !(0.0..=1.0).contains(&keep)
        {
            return Err("the share of quality kept must be within [0, 1]");
        }
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
        let cells = polyhedron::polyhedra(
            lattice.cells().into_iter().map(|(corner, _)| corner),
            &signs,
        )?;
        if cells.is_empty() {
            return Err("no cell of the lattice has a corner inside the surface");
        }
        let points = split::placements(Some(self), &cells, &signs, placement)?;
        let draw = match finish {
            Finish::Draw(keep) => Some((self, keep)),
            _ => None,
        };
        let mut mesh = split::hexahedra(cells, &points, draw)?;
        if let Finish::Fit(freedom, fitting) = finish {
            mesh.inflate(self, freedom, fitting)?;
        }
        Ok(mesh)
    }
}

const SHIFTS: [[Scalar; D]; 5] = [
    [0.0, 0.0, 0.0],
    [0.013_717, 0.007_193, 0.002_971],
    [0.041_351, 0.023_887, 0.011_729],
    [0.097_153, 0.061_771, 0.033_413],
    [0.187_411, 0.140_412, 0.092_153],
];
