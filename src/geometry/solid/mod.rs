//! The [`Solid`] abstraction and the octree → dual → trim → fit meshing driver
//! shared by every solid representation (B-rep, CSG primitives, ...).

#[cfg(test)]
mod test;

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
use std::{
    array::from_fn,
    collections::{VecDeque, hash_map::Entry, HashMap},
    num::NonZeroU32,
    thread::{available_parallelism, scope},
};

const D: usize = 3;

/// Tong et al. 2024 §2.3: keep a hex iff `f_min + TRIM_RATIO * f_max >= 0` over
/// its eight corner signed distances. Mirrors `mesh::tessellation::trim`.
const TRIM_RATIO: Scalar = 0.1;

type Cube = Orthotree<D, 4, 6, 8, u16, NonZeroU32>;

/// A scalar target-element-size field, sampled at octree cell centres.
/// `Sync` so a level of the octree can be tested across threads.
pub trait Sizing: Sync {
    /// The target element size at `point`.
    fn at(&self, point: &Coordinate<D>) -> Quantity<Length>;

    /// The target element size for a cubic cell centred at `center` with
    /// half-edge `half`. The default is [`at`](Self::at) at the centre; a
    /// surface-anchored proximity field overrides it so a thin feature crossing
    /// the cell drives a split even when it misses the cell's centre.
    fn at_cell(&self, center: &Coordinate<D>, _half: Scalar) -> Quantity<Length> {
        self.at(center)
    }
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

/// `oracle.signed_distance` at every coordinate, evaluated across threads
/// (a B-rep oracle ray-casts, so this is the expensive part of classifying).
/// A `false` in `mask` leaves that entry at `NEG_INFINITY`.
fn signed_distances<O: SolidOracle>(
    oracle: &O,
    coordinates: &Coordinates<D>,
    mask: Option<&[bool]>,
) -> Vec<Scalar> {
    let count = coordinates.len();
    let mut signed = vec![Scalar::NEG_INFINITY; count];
    let threads = available_parallelism().map_or(1, |threads| threads.get());
    let chunk = count.div_ceil(threads).max(1);
    scope(|scope| {
        signed
            .chunks_mut(chunk)
            .enumerate()
            .for_each(|(index, out)| {
                scope.spawn(move || {
                    let base = index * chunk;
                    out.iter_mut().enumerate().for_each(|(local, value)| {
                        let node = base + local;
                        if mask.is_none_or(|mask| mask[node]) {
                            *value = oracle.signed_distance(&coordinates[node]);
                        }
                    });
                });
            });
    });
    signed
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

/// The six quad faces of a hexahedron, as corner indices into its node octet.
const HEX_FACES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
];

/// `Inside` / `Cut` / `Outside` per cell by seeded flood fill: a cell whose
/// corner signed distances straddle zero is `Cut` and blocks the fill; the
/// remaining cells are `Outside` if reachable through shared faces from a cell
/// on the mesh boundary, else `Inside`.
///
/// Robust where the oracle's sign is only trustworthy away from creases (a
/// nearest-face-normal B-rep oracle): the fill is seeded both from the mesh
/// rim and from any cell whose eight corners are unanimously outside — a
/// signal even the flaky oracle gets right — so a narrow cavity the `Cut`
/// band seals off at its mouth still drains. Only cells the fill cannot
/// reach, and that are not themselves unanimously outside, end up `Inside`.
/// A fully enclosed void reads `Inside`; single-shell B-reps have none.
pub(crate) fn classify_by_flood_fill(
    oracle: &impl SolidOracle,
    mesh: &Mesh<D>,
) -> Result<Vec<Class>, &'static str> {
    let [Connectivity::Hexahedral(block)] = mesh.connectivities() else {
        return Err("flood-fill classify needs a single hexahedral block");
    };
    let count = block.iter().len();
    let coordinates = mesh.coordinates();
    let signed = signed_distances(oracle, coordinates, None);

    let mut classes = vec![Class::Inside; count];
    let mut cut = vec![false; count];
    for (index, hex) in block.iter().enumerate() {
        let (minimum, maximum) = hex.iter().fold(
            (Scalar::INFINITY, Scalar::NEG_INFINITY),
            |(minimum, maximum), &node| (minimum.min(signed[node]), maximum.max(signed[node])),
        );
        if minimum <= 0.0 && maximum >= 0.0 {
            cut[index] = true;
            classes[index] = Class::Cut;
        }
    }

    // Face adjacency: a quad shared by two hexes links them.
    let mut seen: HashMap<[usize; 4], usize> = HashMap::new();
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); count];
    for (index, hex) in block.iter().enumerate() {
        for face in HEX_FACES {
            let mut key = face.map(|corner| hex[corner]);
            key.sort_unstable();
            match seen.entry(key) {
                Entry::Occupied(slot) => {
                    let other = slot.remove();
                    adjacency[index].push(other);
                    adjacency[other].push(index);
                }
                Entry::Vacant(slot) => {
                    slot.insert(index);
                }
            }
        }
    }

    // Flood `Outside` from every non-`Cut` cell touching the mesh's bounding box.
    let mut low = [Scalar::INFINITY; D];
    let mut high = [Scalar::NEG_INFINITY; D];
    for point in coordinates.iter() {
        for k in 0..D {
            low[k] = low[k].min(point[k].value());
            high[k] = high[k].max(point[k].value());
        }
    }
    let tolerance = (0..D).map(|k| high[k] - low[k]).fold(0.0, Scalar::max) * 1.0e-9;
    let on_boundary = |node: usize| {
        (0..D).any(|k| {
            let value = coordinates[node][k].value();
            value <= low[k] + tolerance || value >= high[k] - tolerance
        })
    };

    // Seed `Outside` from every non-`Cut` cell that either touches the padded
    // octree's rim (air, by construction) or has all eight corners strictly
    // outside (unambiguous even for a crease-flaky oracle). The latter reaches
    // into a cavity the `Cut` band would otherwise wall off at its opening.
    let mut outside = vec![false; count];
    let mut queue = VecDeque::new();
    for (index, hex) in block.iter().enumerate() {
        if cut[index] {
            continue;
        }
        let rim = hex.iter().any(|&node| on_boundary(node));
        let all_outside = hex.iter().all(|&node| signed[node] < 0.0);
        if rim || all_outside {
            outside[index] = true;
            queue.push_back(index);
        }
    }
    while let Some(index) = queue.pop_front() {
        for &neighbour in &adjacency[index] {
            if !cut[neighbour] && !outside[neighbour] {
                outside[neighbour] = true;
                queue.push_back(neighbour);
            }
        }
    }

    for index in 0..count {
        if !cut[index] && outside[index] {
            classes[index] = Class::Outside;
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
    /// `max_levels` (`Some(1..=15)`) caps the octree depth; `None` lets the
    /// sizing field refine as far as the tree allows. `padding` grows the box
    /// by that fraction on each side.
    fn sizing_octree(
        &self,
        sizing: &impl Sizing,
        max_levels: Option<u32>,
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
        max_levels: Option<u32>,
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
        max_levels: Option<u32>,
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
        max_levels: Option<u32>,
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
        let signed = signed_distances(&oracle, mesh.coordinates(), Some(&needed));
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
/// larger than `sizing` allows at its centre. `max_levels` (`Some(1..=15)`)
/// caps the depth; `None` refines as far as the tree allows. `padding` grows
/// the box by that fraction per side.
fn refine_octree(
    (low, high): (Coordinate<D>, Coordinate<D>),
    sizing: &impl Sizing,
    max_levels: Option<u32>,
    padding: Scalar,
) -> Result<Cube, &'static str> {
    // `None` = as deep as the linear octree's u16 corner coordinates allow, so
    // the sizing field alone decides where refinement stops.
    let max_levels = match max_levels {
        None => 15,
        Some(levels) if (1..=15).contains(&levels) => levels,
        Some(_) => return Err("max_levels must be in 1..=15"),
    };
    let low: [Scalar; D] = from_fn(|axis| low[axis].value());
    let high: [Scalar; D] = from_fn(|axis| high[axis].value());
    let root_cells = 1u16 << max_levels;
    let side = (0..D)
        .map(|axis| high[axis] - low[axis])
        .fold(0.0, Scalar::max)
        * (1.0 + padding.max(0.0));
    if side <= 0.0 {
        return Err("degenerate bounding box");
    }
    let cell = Quantity::<Length>::new(side / Scalar::from(root_cells));
    let half = Scalar::from(root_cells) / 2.0;
    // Snap the root so the world origin lands on a grid plane on every axis:
    // grid lines sit at `center - side/2 + i*cell`, so rounding `center -
    // side/2` to a whole number of finest cells puts a plane through 0. Geometry
    // modelled symmetric about a coordinate plane then refines symmetrically
    // instead of at whatever sub-cell phase the raw bbox centre happened to hit.
    // The shift is up to half a finest cell, which the padding margin usually
    // absorbs; where it does not (small padding, coarse levels) it is clamped to
    // the axis's free margin so the geometry never leaves the root box.
    let center: Coordinate<D> = from_fn(|axis| {
        let raw = 0.5 * (low[axis] + high[axis]);
        let base = raw - 0.5 * side;
        let shift = base - (base / cell.value()).round() * cell.value();
        let margin = 0.5 * (side - (high[axis] - low[axis]));
        raw - shift.clamp(-margin, margin)
    })
    .into();

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
    // Refine level by level: test a whole frontier of leaves against `sizing`
    // across threads (the B-rep field ray-casts, so this dominates), then
    // subdivide the ones that are too coarse in one sequential pass.
    let threads = available_parallelism().map_or(1, |threads| threads.get());
    let mut frontier = vec![0usize];
    while !frontier.is_empty() {
        let mut split = vec![false; frontier.len()];
        let chunk = frontier.len().div_ceil(threads).max(1);
        scope(|scope| {
            let (tree, physical, frontier) = (&tree, &physical, &frontier);
            split.chunks_mut(chunk).enumerate().for_each(|(block, out)| {
                scope.spawn(move || {
                    let base = block * chunk;
                    out.iter_mut().enumerate().for_each(|(local, coarse)| {
                        let node = &tree.nodes[frontier[base + local]];
                        let extent = cell * Scalar::from(node.length);
                        let half = 0.5 * extent.value();
                        *coarse = node.length > 1
                            && extent > sizing.at_cell(&physical(node.center()), half);
                    });
                });
            });
        });
        let mut next = Vec::new();
        for (&index, &coarse) in frontier.iter().zip(&split) {
            if !coarse {
                continue;
            }
            tree.subdivide(index)?;
            next.extend(
                tree.nodes[index]
                    .orthants()
                    .expect("subdivided node has orthants")
                    .iter()
                    .map(|slot| slot.slot()),
            );
        }
        frontier = next;
    }
    Ok(tree)
}
