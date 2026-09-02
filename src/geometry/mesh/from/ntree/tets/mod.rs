#[cfg(test)]
mod test;

use super::facets::{Facets, corner_length, leaves};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            from::{kuhn, orient},
        },
        ntree::{
            Octree,
            node::{cell::Cell, slot::Slot},
        },
    },
    math::{FxHashMap, Scalar, Tensor, TensorVec},
};
use std::array::from_fn;

/// The twelve cube edges, each the axis it runs along and the corner it runs
/// from, whose bit for that axis is clear.
const CUBE_EDGES: [(usize, usize); 12] = [
    (0, 0),
    (0, 2),
    (0, 4),
    (0, 6),
    (1, 0),
    (1, 1),
    (1, 4),
    (1, 5),
    (2, 0),
    (2, 1),
    (2, 2),
    (2, 3),
];

/// Splits a cell by [`kuhn`] and then cuts the two tetrahedra sharing the cube
/// edge `from`-`to` at the hanging node on it, giving eight.
///
/// Cutting a tetrahedron at a point on one of its edges cuts each of its faces
/// holding that edge into two about the same point, which is what
/// [`Builder::triangles`] does to the two facets holding this one, so the cell
/// still meets its neighbors as they expect.
fn kuhn_cut(corners: &[usize; 8], from: usize, to: usize, hanging: usize) -> Vec<[usize; 4]> {
    kuhn(&from_fn(|corner| corner))
        .into_iter()
        .flat_map(|tet| {
            if tet.contains(&from) && tet.contains(&to) {
                [from, to]
                    .map(|cut| {
                        tet.map(|corner| {
                            if corner == cut {
                                hanging
                            } else {
                                corners[corner]
                            }
                        })
                    })
                    .to_vec()
            } else {
                vec![tet.map(|corner| corners[corner])]
            }
        })
        .collect()
}

/// Minimal interior tetrahedralisations of the config-A cell — one refined
/// facet, a hanging node on each of its four edges — indexed by which facet is
/// refined.
///
/// Vertices are the cell's own halved lattice offsets, `r0 + 3 r1 + 9 r2` for
/// components 0, 1 and 2, covering the eight corners, the four hanging nodes
/// and the refined facet's centre. Each was found by an exhaustive search over
/// those thirteen nodes against the boundary [`Builder::triangles`] gives that
/// orientation, and is minimal: no tetrahedralisation of fewer than fourteen
/// exists, and none needs an interior node.
///
/// There is a table per orientation rather than one canonical template mapped
/// by symmetry because the triangulation rules order nodes lexicographically,
/// which is absolute, not relative to the cell; rotating a template would
/// carry a diagonal onto a face whose own rule puts it elsewhere.
const CONFIG_A: [[[usize; 4]; 14]; 6] = [
    [
        [0, 20, 2, 26],
        [0, 26, 2, 8],
        [0, 8, 3, 26],
        [0, 26, 3, 12],
        [0, 12, 9, 26],
        [0, 26, 9, 20],
        [3, 8, 6, 26],
        [3, 26, 6, 15],
        [3, 15, 12, 26],
        [9, 26, 12, 21],
        [9, 21, 18, 26],
        [9, 26, 18, 20],
        [12, 26, 15, 24],
        [12, 24, 21, 26],
    ],
    [
        [0, 11, 2, 14],
        [0, 14, 2, 5],
        [0, 14, 5, 17],
        [0, 17, 5, 8],
        [0, 8, 6, 17],
        [0, 17, 6, 26],
        [0, 26, 6, 24],
        [0, 20, 11, 23],
        [0, 23, 11, 14],
        [0, 23, 14, 26],
        [0, 26, 14, 17],
        [0, 24, 18, 26],
        [0, 26, 18, 23],
        [0, 23, 18, 20],
    ],
    [
        [0, 10, 1, 26],
        [0, 26, 1, 8],
        [0, 8, 6, 26],
        [0, 26, 6, 24],
        [0, 24, 9, 26],
        [0, 26, 9, 10],
        [1, 11, 2, 26],
        [1, 26, 2, 8],
        [1, 26, 10, 11],
        [9, 19, 10, 26],
        [9, 24, 18, 26],
        [9, 26, 18, 19],
        [10, 20, 11, 26],
        [10, 26, 19, 20],
    ],
    [
        [0, 20, 2, 26],
        [0, 26, 2, 17],
        [0, 17, 2, 8],
        [0, 7, 6, 16],
        [0, 16, 6, 15],
        [0, 8, 7, 17],
        [0, 17, 7, 16],
        [0, 16, 15, 25],
        [0, 25, 15, 24],
        [0, 17, 16, 26],
        [0, 26, 16, 25],
        [0, 24, 18, 25],
        [0, 25, 18, 26],
        [0, 26, 18, 20],
    ],
    [
        [0, 20, 1, 26],
        [0, 26, 1, 4],
        [0, 4, 3, 26],
        [0, 26, 3, 24],
        [0, 24, 18, 26],
        [0, 26, 18, 20],
        [1, 20, 2, 26],
        [1, 26, 2, 5],
        [1, 5, 4, 26],
        [3, 26, 4, 7],
        [3, 7, 6, 26],
        [3, 26, 6, 24],
        [4, 26, 5, 8],
        [4, 8, 7, 26],
    ],
    [
        [0, 20, 2, 23],
        [0, 23, 2, 26],
        [0, 26, 2, 8],
        [0, 8, 6, 26],
        [0, 26, 6, 25],
        [0, 25, 6, 24],
        [0, 21, 18, 22],
        [0, 22, 18, 19],
        [0, 22, 19, 23],
        [0, 23, 19, 20],
        [0, 24, 21, 25],
        [0, 25, 21, 22],
        [0, 25, 22, 26],
        [0, 26, 22, 23],
    ],
];

/// Whether the hanging edges are exactly the four bounding `facet`.
fn on_facet(hanging: &[(usize, usize)], facet: usize) -> bool {
    let (axis, side) = (facet >> 1, facet & 1);
    hanging.iter().all(|&(edge, _)| {
        let (along, from) = CUBE_EDGES[edge];
        along != axis && (from >> axis) & 1 == side
    })
}

/// The config-A template for `refined`, resolved to this cell's nodes.
fn config_a(
    facets: &Facets<3>,
    corner: [usize; 3],
    length: usize,
    refined: usize,
) -> Option<Vec<[usize; 4]>> {
    let half = length / 2;
    let mut tets = Vec::with_capacity(CONFIG_A[refined].len());
    for template in &CONFIG_A[refined] {
        let mut tet = [0; 4];
        for (slot, &vertex) in tet.iter_mut().zip(template.iter()) {
            let offsets = [vertex % 3, vertex / 3 % 3, vertex / 9];
            *slot = facets.node(&from_fn(|axis| corner[axis] + offsets[axis] * half))?
        }
        tets.push(tet)
    }
    Some(tets)
}

/// Lexicographic rank of a node, most significant axis first, so that a cell's
/// extremes under it are the corners [`kuhn`] splits about.
fn rank(key: [usize; 3]) -> [usize; 3] {
    [key[2], key[1], key[0]]
}

/// Builds the tetrahedra of an octree, holding the nodes added beyond the leaf
/// corners: the centres of the facet patches that carry hanging nodes, and the
/// centroid of every cell filled from a template.
struct Builder<'a> {
    added: FxHashMap<[usize; 3], usize>,
    corners: usize,
    extra: Coordinates<3>,
    facets: &'a Facets<3>,
}

impl<'a> Builder<'a> {
    fn new(facets: &'a Facets<3>) -> Self {
        Self {
            added: FxHashMap::default(),
            corners: facets.coordinates().len(),
            extra: Coordinates::new(),
            facets,
        }
    }
    /// The node at a point given in doubled integer lattice coordinates, so
    /// that cell and facet centres land on exact keys and any two cells naming
    /// the same point get the same node.
    fn node(&mut self, doubled: [usize; 3]) -> usize {
        if doubled.iter().all(|component| component % 2 == 0)
            && let Some(node) = self.facets.node(&from_fn(|axis| doubled[axis] / 2))
        {
            return node;
        }
        if let Some(&node) = self.added.get(&doubled) {
            return node;
        }
        let node = self.corners + self.extra.len();
        self.extra.push(Coordinate::from(from_fn(|axis| {
            doubled[axis] as Scalar / 2.0
        })));
        self.added.insert(doubled, node);
        node
    }
    fn centre(&mut self, polygon: &[usize]) -> usize {
        let mut low = [usize::MAX; 3];
        let mut high = [0; 3];
        polygon.iter().for_each(|&node| {
            let key = self.facets.key(node);
            (0..3).for_each(|axis| {
                low[axis] = low[axis].min(key[axis]);
                high[axis] = high[axis].max(key[axis])
            })
        });
        self.node(from_fn(|axis| low[axis] + high[axis]))
    }
    /// Triangulates one facet patch as a function of its own nodes alone.
    ///
    /// A bare square is cut by the diagonal between its lexicographic extremes,
    /// which is the diagonal [`kuhn`] induces on it. One hanging node splits
    /// whichever of those two triangles holds it, so that a cell meshed by
    /// [`kuhn`] and then cut along that edge induces exactly this. Two or more
    /// hanging nodes fan from the smallest of them: the patch is convex, so a
    /// fan from any vertex covers it, and only the vertices on a hanging
    /// node's own edge are collinear with it while that edge's two corners are
    /// never adjacent in the fan, so no triangle is degenerate.
    ///
    /// All of that assumes at most one hanging node to an edge, which 2:1
    /// balancing gives; the centre fan is the total fallback for the rest.
    ///
    /// Every rule reads the patch alone, so the cells either side of it agree,
    /// and all are invariant under reversing the loop, so their opposite
    /// windings cut it the same way.
    fn triangles(&mut self, polygon: &[usize]) -> Vec<[usize; 3]> {
        let count = polygon.len();
        let keys: Vec<[usize; 3]> = polygon.iter().map(|&node| self.facets.key(node)).collect();
        let mut low = [usize::MAX; 3];
        let mut high = [0; 3];
        keys.iter().for_each(|key| {
            (0..3).for_each(|axis| {
                low[axis] = low[axis].min(key[axis]);
                high[axis] = high[axis].max(key[axis])
            })
        });
        let corner = |vertex: usize| {
            (0..3).all(|axis| [low[axis], high[axis]].contains(&keys[vertex][axis]))
        };
        let corners: Vec<usize> = (0..count).filter(|&vertex| corner(vertex)).collect();
        let crowded = (0..count).any(|vertex| !corner(vertex) && !corner((vertex + 1) % count));
        if corners.len() != 4 || crowded {
            let centre = self.centre(polygon);
            return (0..count)
                .map(|k| [centre, polygon[k], polygon[(k + 1) % count]])
                .collect();
        }
        if count == 5 {
            // The lone hanging node lies in the gap after corner `gap`.
            let gap = (0..4)
                .find(|&k| (corners[k] + 1) % count != corners[(k + 1) % 4])
                .expect("no hanging node");
            let start = (0..4)
                .min_by_key(|&k| rank(keys[corners[k]]))
                .expect("no corner");
            let quad: [usize; 4] = from_fn(|k| polygon[corners[(start + k) % 4]]);
            let [a, b, c, d] = quad;
            let hanging = polygon[(corners[gap] + 1) % count];
            return match (gap + 4 - start) % 4 {
                0 => vec![[a, hanging, c], [hanging, b, c], [a, c, d]],
                1 => vec![[a, b, hanging], [a, hanging, c], [a, c, d]],
                2 => vec![[a, b, c], [a, c, hanging], [a, hanging, d]],
                _ => vec![[a, b, c], [a, c, hanging], [hanging, c, d]],
            };
        }
        let pivot = (0..count)
            .filter(|&vertex| count == 4 || !corner(vertex))
            .min_by_key(|&vertex| rank(keys[vertex]))
            .expect("no pivot");
        (1..count - 1)
            .map(|k| {
                [
                    polygon[pivot],
                    polygon[(pivot + k) % count],
                    polygon[(pivot + k + 1) % count],
                ]
            })
            .collect()
    }
}

impl Mesh<3> {
    pub(crate) fn tetrahedra_from<T, U, V>(octree: Octree<T, U, V>) -> Self
    where
        T: Cell,
        U: Slot,
    {
        Self::tetrahedra(octree).0
    }
    /// Meshes an octree as tetrahedra, with the number of leaves taken by each
    /// branch.
    ///
    /// Requires a 2:1 balanced tree: a leaf facing a neighbor more than one
    /// level finer is not matched against all of that neighbor's refinement.
    ///
    /// A leaf whose six facets are bare squares is split into six by
    /// [`kuhn`]. Any other leaf — one meeting finer neighbors, or carrying a
    /// hanging node on a facet edge — has its facets triangulated by the rules
    /// in [`Builder::triangles`] and is filled by coning those triangles to its
    /// own centroid. The cell is convex and the triangulation of its boundary
    /// closed, so that fill is positive and its interior faces pair off inside
    /// the cell; the facet triangles pair with the neighbor's because both
    /// sides derive them from the same nodes.
    pub(crate) fn tetrahedra<T, U, V>(octree: Octree<T, U, V>) -> (Self, usize, usize)
    where
        T: Cell,
        U: Slot,
    {
        let (leaves, _) = leaves(&octree);
        let facets = Facets::<3>::new(&octree, &leaves);
        let mut builder = Builder::new(&facets);
        let mut connectivity = Vec::with_capacity(6 * leaves.len());
        let (mut plain, mut templated) = (0, 0);
        leaves.iter().for_each(|&index| {
            let (corner, length) = corner_length(&octree.nodes[index]);
            let polygons = facets.leaf_polygons(&octree, index);
            if polygons
                .iter()
                .all(|facet| facet.len() == 1 && facet[0].len() == 4)
            {
                plain += 1;
                connectivity.extend(kuhn(&facets.corners::<8>(corner, length)))
            } else {
                templated += 1;
                let mut hanging = Vec::new();
                if length % 2 == 0 {
                    CUBE_EDGES
                        .iter()
                        .enumerate()
                        .for_each(|(edge, &(axis, low))| {
                            let mut key: [usize; 3] =
                                from_fn(|a| corner[a] + ((low >> a) & 1) * length);
                            key[axis] += length / 2;
                            if let Some(node) = facets.node(&key) {
                                hanging.push((edge, node))
                            }
                        })
                }
                let refined: Vec<usize> = (0..6).filter(|&f| polygons[f].len() > 1).collect();
                let template = match refined[..] {
                    [facet] if hanging.len() == 4 && on_facet(&hanging, facet) => {
                        config_a(&facets, corner, length, facet)
                    }
                    _ => None,
                };
                match (template, hanging.first()) {
                    (Some(tets), _) => connectivity.extend(tets),
                    (None, Some(&(edge, node))) if hanging.len() == 1 && refined.is_empty() => {
                        let (axis, low) = CUBE_EDGES[edge];
                        connectivity.extend(kuhn_cut(
                            &facets.corners::<8>(corner, length),
                            low,
                            low | (1 << axis),
                            node,
                        ))
                    }
                    _ => {
                        let apex = builder.node(from_fn(|axis| 2 * corner[axis] + length));
                        polygons.iter().flatten().for_each(|polygon| {
                            connectivity.extend(
                                builder
                                    .triangles(polygon)
                                    .into_iter()
                                    .map(|[a, b, c]| [a, b, c, apex]),
                            )
                        })
                    }
                }
            }
        });
        let extra = builder.extra;
        let mut coordinates = facets.into_coordinates();
        extra
            .into_iter()
            .for_each(|coordinate| coordinates.push(coordinate));
        octree.rescale_coordinates(&mut coordinates);
        orient(&mut connectivity, &coordinates);
        (
            (
                vec![Connectivity::Tetrahedral(connectivity.into())],
                coordinates,
            )
                .into(),
            plain,
            templated,
        )
    }
}
