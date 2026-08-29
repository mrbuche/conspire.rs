//! The [`Solid`] abstraction and the octree → dual → trim → fit meshing driver
//! shared by every solid representation (B-rep, CSG primitives, ...).

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction,
        mesh::{Class, Connectivity, Fitting, Mesh, buffer::fit::Oracle},
        ntree::{
            Balance, Balancing, Dualization, Orthotree, Pairing, Rescaling,
            node::{Kind, Node, slot::Slot},
        },
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::{array::from_fn, num::NonZeroU32};

const D: usize = 3;

/// Tong et al. 2024 §2.3: keep a hex iff `f_min + TRIM_RATIO * f_max >= 0` over
/// its eight corner signed distances. Mirrors `mesh::tessellation::trim`.
const TRIM_RATIO: Scalar = 0.1;

type Cube = Orthotree<D, 4, 6, 8, u16, NonZeroU32>;

/// A scalar target-element-size field, sampled at octree cell centres.
pub trait Sizing {
    /// The target element size at `point`.
    fn at(&self, point: &Coordinate<D>) -> Quantity<Length>;
}

/// A constant target element size everywhere.
pub struct Uniform(pub Quantity<Length>);

impl Sizing for Uniform {
    fn at(&self, _point: &Coordinate<D>) -> Quantity<Length> {
        self.0
    }
}

/// An analytic query surface for the boundary fit: closest-point projection and
/// signed distance, the latter positive inside the solid.
pub trait SolidOracle: Sync {
    /// The closest surface point to `query` and the outward unit normal there.
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)>;
    /// Signed distance from `query` to the surface, positive inside the solid.
    fn signed_distance(&self, query: &Coordinate<D>) -> Scalar;
}

/// `Inside` / `Cut` / `Outside` per cell of `mesh` from an oracle's signed
/// distance: a cell whose corner distances straddle zero is `Cut`, else its
/// centroid's sign decides.
pub(crate) fn classify_by_signed_distance(
    oracle: &impl SolidOracle,
    mesh: &Mesh<D>,
) -> Result<Vec<Class>, &'static str> {
    let signed: Vec<Scalar> = mesh
        .coordinates()
        .iter()
        .map(|point| oracle.signed_distance(point))
        .collect();
    let centroids = mesh.centroids();
    let mut classes = Vec::with_capacity(mesh.number_of_elements());
    let mut index = 0;
    for block in mesh.iter() {
        for element in block.iter() {
            let (minimum, maximum) = block.element_nodes(element).iter().fold(
                (Scalar::INFINITY, Scalar::NEG_INFINITY),
                |(minimum, maximum), &node| (minimum.min(signed[node]), maximum.max(signed[node])),
            );
            classes.push(if minimum <= 0.0 && maximum >= 0.0 {
                Class::Cut
            } else if oracle.signed_distance(&centroids[index]) > 0.0 {
                Class::Inside
            } else {
                Class::Outside
            });
            index += 1;
        }
    }
    Ok(classes)
}

/// Bridges a [`SolidOracle`] to the buffer layer's private fit [`Oracle`].
struct Fit<'a, O>(&'a O);

impl<O: SolidOracle> Oracle for Fit<'_, O> {
    fn project(&self, query: &Coordinate<D>) -> Option<(Coordinate<D>, Direction<D>)> {
        self.0.project(query)
    }
}

/// A solid the shared driver can mesh: a bounding box, a classifier against a
/// background mesh, and an analytic oracle for the boundary fit.
pub trait Solid {
    type Oracle: SolidOracle;

    /// The `(low, high)` corners of an axis-aligned box enclosing the solid.
    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str>;

    /// An analytic oracle projecting onto the exact surface.
    fn oracle(&self) -> Result<Self::Oracle, &'static str>;

    /// Labels every cell of `mesh` `Inside`, `Cut`, or `Outside`. The default
    /// reads the [`oracle`](Self::oracle): a cell whose corner signed distances
    /// straddle zero is `Cut`, otherwise its centroid's sign decides.
    fn classify(&self, mesh: &Mesh<D>) -> Result<Vec<Class>, &'static str> {
        classify_by_signed_distance(&self.oracle()?, mesh)
    }

    /// Refines an octree over the padded bounding box until every leaf is no
    /// larger than `sizing` allows at its centre, then returns the leaves as a
    /// hexahedral mesh: a graded block of cubes, hanging nodes and all.
    ///
    /// `max_levels` (1..=15) caps the octree depth; `padding` grows the box by
    /// that fraction on each side.
    fn sizing_octree(
        &self,
        sizing: &impl Sizing,
        max_levels: u32,
        padding: Scalar,
    ) -> Result<Mesh<D>, &'static str> {
        let tree = refine_octree(self.bounding_box()?, sizing, max_levels, padding)?;
        let rescale = Rescaling {
            center: tree.rescale().center.clone(),
            cell: tree.rescale().cell,
            half: tree.rescale().half,
        };
        let (connectivity, mut coordinates): (Vec<[usize; 8]>, Coordinates<D>) = tree.into();
        coordinates
            .iter_mut()
            .for_each(|coordinate| *coordinate = rescale.apply(coordinate));
        Ok((
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into())
    }

    /// Refines the octree from `sizing`, balances and dualizes it, and
    /// classifies the resulting conforming all-hex mesh against this solid.
    /// `balancing` must be `Strong(1)` or `Weak(1)`.
    fn dual_background(
        &self,
        sizing: &impl Sizing,
        max_levels: u32,
        padding: Scalar,
        balancing: Balancing,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let mut tree = refine_octree(self.bounding_box()?, sizing, max_levels, padding)?;
        tree.equilibrate(balancing, Pairing::Regular)?;
        let mesh = tree.dualize();
        let classes = self.classify(&mesh)?;
        Ok((mesh, classes))
    }

    /// Builds the [`dual_background`](Self::dual_background) and drops the
    /// `Outside` cells. The kept `Cut` cells straddle the surface and come back
    /// for a later fit or cut step, so this is a trimmed background, not a
    /// finished mesh.
    fn trim(
        &self,
        sizing: &impl Sizing,
        max_levels: u32,
        padding: Scalar,
        balancing: Balancing,
    ) -> Result<(Mesh<D>, Vec<Class>), &'static str> {
        let (mut mesh, classes) = self.dual_background(sizing, max_levels, padding, balancing)?;
        let keep: Vec<bool> = classes
            .iter()
            .map(|&class| class != Class::Outside)
            .collect();
        mesh.keep_hexes(|index, _, _| keep[index])?;
        let classes = classes
            .into_iter()
            .zip(&keep)
            .filter_map(|(class, &keep)| keep.then_some(class))
            .collect();
        Ok((mesh, classes))
    }

    /// Meshes this solid end to end: refine the octree from `sizing`, balance,
    /// dualize, trim interior cells back to the surface by Tong's per-hex SDF
    /// ratio rule, then inflate a boundary layer and fit it to the exact
    /// geometry with the analytic [`oracle`](Self::oracle).
    fn mesh(
        &self,
        sizing: &impl Sizing,
        max_levels: u32,
        padding: Scalar,
        balancing: Balancing,
        fitting: Fitting,
    ) -> Result<Mesh<D>, &'static str> {
        let (mut mesh, classes) = self.dual_background(sizing, max_levels, padding, balancing)?;
        let oracle = self.oracle()?;
        let outside: Vec<bool> = classes
            .iter()
            .map(|&class| class == Class::Outside)
            .collect();
        let number_of_nodes = mesh.coordinates().len();
        let mut needed = vec![false; number_of_nodes];
        {
            let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
                return Err("dual background is not a single hexahedral block");
            };
            block.iter().zip(&outside).for_each(|(hex, &out)| {
                if !out {
                    hex.iter().for_each(|&node| needed[node] = true)
                }
            });
        }
        let signed: Vec<Scalar> = (0..number_of_nodes)
            .map(|node| {
                if needed[node] {
                    oracle.signed_distance(&mesh.coordinates()[node])
                } else {
                    Scalar::NEG_INFINITY
                }
            })
            .collect();
        mesh.keep_hexes(|index, hex, _| {
            if outside[index] {
                return false;
            }
            let (minimum, maximum) = hex.iter().fold(
                (Scalar::INFINITY, Scalar::NEG_INFINITY),
                |(minimum, maximum), &node| (minimum.min(signed[node]), maximum.max(signed[node])),
            );
            minimum + TRIM_RATIO * maximum >= 0.0
        })?;
        mesh.buffer_with(&Fit(&oracle), fitting)
    }
}

/// Refines an octree over the padded box `(low, high)` until every leaf is no
/// larger than `sizing` allows at its centre. `max_levels` (1..=15) caps the
/// depth; `padding` grows the box by that fraction per side.
fn refine_octree(
    (low, high): (Coordinate<D>, Coordinate<D>),
    sizing: &impl Sizing,
    max_levels: u32,
    padding: Scalar,
) -> Result<Cube, &'static str> {
    if !(1..=15).contains(&max_levels) {
        return Err("max_levels must be in 1..=15");
    }
    let low: [Scalar; D] = from_fn(|axis| low[axis].value());
    let high: [Scalar; D] = from_fn(|axis| high[axis].value());
    let root_cells = 1u16 << max_levels;
    let center: Coordinate<D> = from_fn(|axis| 0.5 * (low[axis] + high[axis])).into();
    let side = (0..D)
        .map(|axis| high[axis] - low[axis])
        .fold(0.0, Scalar::max)
        * (1.0 + padding.max(0.0));
    if side <= 0.0 {
        return Err("degenerate bounding box");
    }
    let cell = Quantity::<Length>::new(side / Scalar::from(root_cells));
    let half = Scalar::from(root_cells) / 2.0;

    let mut tree: Cube = Orthotree {
        balanced: Balancing::None,
        paired: Pairing::None,
        rescale: Rescaling {
            center: center.clone(),
            cell,
            half,
        },
        nodes: vec![Node {
            corner: [0; D],
            length: root_cells,
            facets: [None; 6],
            kind: Kind::Leaf,
            value: None,
        }],
    };
    let physical = |cells: [u16; D]| -> Coordinate<D> {
        from_fn(|axis| cell.value() * (Scalar::from(cells[axis]) - half) + center[axis].value())
            .into()
    };
    let mut stack = vec![0usize];
    while let Some(index) = stack.pop() {
        let length = tree.nodes[index].length;
        if length <= 1 {
            continue;
        }
        let extent = cell * Scalar::from(length);
        if extent <= sizing.at(&physical(tree.nodes[index].center())) {
            continue;
        }
        tree.subdivide(index)?;
        let children: Vec<usize> = tree.nodes[index]
            .orthants()
            .expect("subdivided node has orthants")
            .iter()
            .map(|slot| slot.slot())
            .collect();
        stack.extend(children);
    }
    Ok(tree)
}
