#[cfg(test)]
mod test;

mod assemble;
mod build;
mod classify;
mod cleanup;
mod face;
mod geometry;
mod lattice;
mod snap;
mod split;
mod tables;
mod topology;

use crate::{
    geometry::{
        Coordinate, Direction,
        mesh::{
            Mesh,
            tessellation::{D, Tessellation, cut::geometry::contained},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing, Sizing},
    },
    math::{Quantity, Scalar},
    units::Length,
};
use std::{collections::HashMap, num::NonZeroU32};

const COLLAPSE_FRACTION: Scalar = 0.2;
const CROSSING_TOLERANCE: Quantity<Length> = Length::meters(1.0e-8);
const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
const SLIVER_FRACTION: Scalar = 0.1;
const SNAP_FEATURE: Scalar = 0.5;
const SNAP_HARD: Scalar = 0.05;
const SNAP_QUALITY: Scalar = 0.3;
const SNAP_SOFT: Scalar = 0.2;
const FACES: [[usize; 4]; 6] = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 3, 2, 1],
    [4, 5, 6, 7],
];
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
const DIRECTIONS: [Direction<D>; 3] = [
    Direction::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Direction::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Direction::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

/// What an octree background's cells are meshed into.
enum Cells {
    Polyhedral,
    Tetrahedral,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    Inside,
    Cut,
    Outside,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sign {
    Inside,
    On,
    Outside,
}

/// Identifies a point used while stitching a cut face/cell.
///
/// Either an original mesh node, or the `ordinal`-th crossing,
/// (in canonical ascending-node-order direction) along the sorted edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Vertex {
    Node(usize),
    Crossing([usize; 2], usize),
}

pub struct Tables {
    signs: HashMap<usize, Sign>,
    crossings: HashMap<[usize; 2], Vec<Coordinate<D>>>,
    faces: HashMap<[usize; 4], [usize; 4]>,
    segments: HashMap<[usize; 4], Vec<[Vertex; 2]>>,
}

impl Tables {
    pub fn signs(&self) -> &HashMap<usize, Sign> {
        &self.signs
    }
    pub fn crossings(&self) -> &HashMap<[usize; 2], Vec<Coordinate<D>>> {
        &self.crossings
    }
    pub fn faces(&self) -> &HashMap<[usize; 4], [usize; 4]> {
        &self.faces
    }
    pub fn segments(&self) -> &HashMap<[usize; 4], Vec<[Vertex; 2]>> {
        &self.segments
    }
}

impl Tessellation {
    /// Builds the dual of an octree fitted to this tessellation, with each
    /// cell classified against the surface.
    ///
    /// The background for [`cut`](Self::cut). `balancing` must be `Strong(1)`
    /// or `Weak(1)`, which is what dualization requires.
    pub fn dual_background(
        &self,
        balancing: Balancing,
        scale: Scalar,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let sizing = Sizing::new(self, scale, CurvatureSizing::default(), PADDING);
        let mesh = if sizing.fits::<u16>() {
            let mut octree = Octree::<u16, NonZeroU32>::refine(&sizing)?;
            octree.equilibrate(balancing, Pairing::Regular)?;
            octree.dualize()
        } else {
            let mut octree = Octree::<u32, NonZeroU32>::refine(&sizing)?;
            octree.equilibrate(balancing, Pairing::Regular)?;
            octree.dualize()
        };
        let classes = self.classify(&mesh);
        Ok((mesh, classes))
    }
    /// Builds a uniform lattice of cubes of the given edge length around this
    /// tessellation, with each cell classified against the surface.
    ///
    /// The lattice spans the cells the surface passes through, those its
    /// interior encloses, and a single shell of cells beyond them, so it is a
    /// background to be [cut](Self::cut), or [trimmed](Self::trim) and
    /// [buffered](Mesh::buffer), rather than a finished mesh.
    ///
    /// Unlike [`dual_background`](Self::dual_background) the cells are all
    /// axis-aligned cubes, at the cost of the grading a tree provides, and
    /// the classes fall out of rasterizing rather than being found again.
    pub fn lattice_background(
        &self,
        spacing: Quantity<Length>,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        Ok(self.lattice_cells(spacing)?.mesh())
    }
    /// Builds a uniform lattice around this tessellation and splits every cell
    /// into six tetrahedra, with each one classified against the surface.
    ///
    /// The tetrahedral counterpart of
    /// [`lattice_background`](Self::lattice_background). The cells are still
    /// classified by rasterizing, so the six tetrahedra of a cell all take the
    /// class of the cell they came from.
    pub fn lattice_tet_background(
        &self,
        spacing: Quantity<Length>,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        Ok(self.lattice_cells(spacing)?.tets())
    }
    /// Builds an octree fitted to this tessellation, with each cell
    /// classified against the surface.
    ///
    /// The background for [`cut_polyhedral`](Self::cut_polyhedral), taking
    /// the octree directly rather than its dual. This places no 2:1
    /// requirement on `balancing`, since hanging nodes become extra vertices
    /// on a face rather than something to be dualized away. `Weak(n)` and
    /// `Strong(n)` for `n > 1` are therefore available here, permitting
    /// coarser trees than dualization allows.
    pub fn octree_background(
        &self,
        balancing: Balancing,
        scale: Scalar,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let mesh = self.octree_mesh(
            balancing,
            Pairing::Regular,
            scale,
            CurvatureSizing::default(),
            Cells::Polyhedral,
        )?;
        let classes = self.classify(&mesh);
        Ok((mesh, classes))
    }
    /// Builds an octree fitted to this tessellation and meshes it as
    /// tetrahedra, with each one classified against the surface.
    ///
    /// The tetrahedral counterpart of
    /// [`octree_background`](Self::octree_background), to be
    /// [trimmed](Self::trim). `balancing` must be `Strong(1)`: the templates
    /// filling a graded cell only span a one-level difference, and only a
    /// balance over edges and vertices as well as faces holds them to it.
    /// `pairing` need not be `Regular`; the tetrahedra conform under any
    /// pairing, and `None` yields a smaller background.
    ///
    /// `tolerance` is the Dunyach chord-error tolerance for curvature-driven
    /// refinement; `None` disables it.
    pub fn octree_tet_background(
        &self,
        balancing: Balancing,
        pairing: Pairing,
        scale: Scalar,
        tolerance: Option<Quantity<Length>>,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        if !matches!(balancing, Balancing::Strong(1)) {
            return Err("tetrahedra require Strong(1) balancing");
        }
        let curvature = CurvatureSizing {
            tolerance,
            ..Default::default()
        };
        let mesh = self.octree_mesh(balancing, pairing, scale, curvature, Cells::Tetrahedral)?;
        let classes = self.classify(&mesh);
        Ok((mesh, classes))
    }
    fn octree_mesh(
        &self,
        balancing: Balancing,
        pairing: Pairing,
        scale: Scalar,
        curvature: CurvatureSizing,
        cells: Cells,
    ) -> Result<Mesh<D>, &'static str> {
        let sizing = Sizing::new(self, scale, curvature, PADDING);
        if sizing.fits::<u16>() {
            let mut octree = Octree::<u16, NonZeroU32>::refine(&sizing)?;
            octree.equilibrate(balancing, pairing)?;
            Ok(match cells {
                Cells::Polyhedral => Mesh::from(octree),
                Cells::Tetrahedral => Mesh::tetrahedra_from(octree),
            })
        } else {
            let mut octree = Octree::<u32, NonZeroU32>::refine(&sizing)?;
            octree.equilibrate(balancing, pairing)?;
            Ok(match cells {
                Cells::Polyhedral => Mesh::from(octree),
                Cells::Tetrahedral => Mesh::tetrahedra_from(octree),
            })
        }
    }
    /// Cuts a classified background mesh to this tessellation, leaving
    /// hexahedra everywhere but at the boundary.
    ///
    /// Snaps the nodes that nearly lie on the surface onto it, builds the
    /// crossing tables, and assembles the cut cells into polyhedra.
    pub fn cut(&self, mesh: Mesh<D>, classes: &[Class]) -> Result<Mesh<D>, &'static str> {
        if !contained(&mesh, classes) {
            return Err("tessellation is not contained within the background mesh");
        }
        let (mesh, snapped) = self.snap(mesh, classes)?;
        let tables = self.tables(&mesh, classes, &snapped)?;
        self.assemble(&mesh, classes, &tables)
    }
    /// Cuts a classified background mesh to this tessellation, leaving
    /// polyhedra throughout.
    ///
    /// The counterpart of [`cut`](Self::cut) for a background whose cells
    /// carry hanging nodes, such as an octree taken directly.
    pub fn cut_polyhedral(
        &self,
        mesh: Mesh<D>,
        classes: &[Class],
    ) -> Result<Mesh<D>, &'static str> {
        if !contained(&mesh, classes) {
            return Err("tessellation is not contained within the background mesh");
        }
        let (mesh, snapped) = self.snap_generic(mesh, classes)?;
        let tables = self.tables_generic(&mesh, classes, &snapped)?;
        self.assemble_generic(&mesh, classes, &tables)
    }
}
