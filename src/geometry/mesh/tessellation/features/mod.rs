#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, Direction,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity,
            tessellation::{D, Tessellation},
        },
    },
    math::{CrossProduct, FxHashMap, FxHashSet, Quantity, Scalar, Tensor},
    units::{Area, Dimensionless, Length},
};
use std::array::from_fn;

/// Cosine of the dihedral deviation past which an edge is a crease.
const CREASE_COSINE: Scalar = 0.866_025_403_784_438_6;

/// How far, as a fraction of a pair's own gap, the surface joining two
/// creases may stray from the straight line between them and still count as
/// joining them (see [`joined`]).
const JOINED_FRACTION: Scalar = 0.25;

/// How many places along a pair of creases the surface between them is
/// sampled (see [`joined`]).
const JOINED_SAMPLES: usize = 5;

/// How much farther than the closest crease of a chain another crease of that
/// same chain may run from a crease and still count as bounding the same
/// narrow feature (see [`Features::separation`]).
const SEPARATION_SPREAD: Scalar = 1.05;

/// The sharp edges and points of a tessellation.
///
/// An edge is a crease when its two triangles turn through more than thirty
/// degrees. A vertex is a corner when the creases through it do not simply
/// pass through: it ends one, joins three or more, or turns sharply between
/// two.
///
/// Only edges with exactly two incident triangles are considered, so a
/// tessellation whose triangles do not share nodes has no features rather
/// than having every edge be one.
pub struct Features {
    corners: Vec<Coordinate<D>>,
    /// The tessellation node underlying each entry in `corners`, kept to cut
    /// the crease graph into chains (see [`Features::separation`]).
    corner_nodes: Vec<usize>,
    creases: Vec<[Coordinate<D>; 2]>,
    /// The tessellation node pair underlying each entry in `creases`, kept to
    /// walk the crease graph for adjacency (see [`Features::separation`]).
    crease_nodes: Vec<[usize; 2]>,
}

/// The nearest topologically unrelated crease to a crease, and how far away
/// it is (see [`Features::separation`]).
#[derive(Clone, Copy, Debug)]
pub struct Separation {
    /// Index into [`Features::creases`] of the crease that was matched.
    pub crease: usize,
    /// The closest-point distance between the two creases.
    pub distance: Quantity<Length>,
}

/// The features of a tessellation, binned for lookup within a fixed radius.
pub struct FeatureIndex<'a> {
    features: &'a Features,
    corners: FxHashMap<[i64; D], Vec<usize>>,
    creases: FxHashMap<[i64; D], Vec<usize>>,
    /// Creases spanning too many cells to bin, scanned by every lookup.
    sprawling: Vec<usize>,
    spacing: Quantity<Length>,
}

/// How many cells a crease may be binned into before it is left sprawling.
const SPAN: i64 = 64;

fn triangles(tessellation: &Tessellation) -> Vec<[usize; D]> {
    match &tessellation.mesh().connectivities()[0] {
        Connectivity::Triangular(triangles) => triangles.iter().copied().collect(),
        _ => Vec::new(),
    }
}

fn key(one: usize, two: usize) -> [usize; 2] {
    if one < two { [one, two] } else { [two, one] }
}

pub(crate) fn crease_edges(
    triangles: &[[usize; D]],
    coordinates: &Coordinates<D>,
) -> Vec<[usize; 2]> {
    let normals: Vec<Direction<D>> = triangles
        .iter()
        .map(|&[a, b, c]| {
            (&coordinates[b] - &coordinates[a])
                .cross(&(&coordinates[c] - &coordinates[a]))
                .normalized()
        })
        .collect();
    let mut incident = FxHashMap::<[usize; 2], Vec<usize>>::default();
    triangles
        .iter()
        .enumerate()
        .for_each(|(index, &[a, b, c])| {
            [key(a, b), key(b, c), key(c, a)]
                .into_iter()
                .for_each(|edge| incident.entry(edge).or_default().push(index))
        });
    let mut sharp: Vec<[usize; 2]> = incident
        .iter()
        .filter(|(_, triangles)| triangles.len() == 2)
        .filter(|(_, triangles)| &normals[triangles[0]] * &normals[triangles[1]] < CREASE_COSINE)
        .map(|(&edge, _)| edge)
        .collect();
    sharp.sort_unstable();
    sharp
}

pub(crate) fn crease_nodes(
    triangles: &[[usize; D]],
    coordinates: &Coordinates<D>,
) -> FxHashSet<usize> {
    crease_edges(triangles, coordinates)
        .into_iter()
        .flatten()
        .collect()
}

impl Features {
    pub fn corners(&self) -> &[Coordinate<D>] {
        &self.corners
    }
    pub fn creases(&self) -> &[[Coordinate<D>; 2]] {
        &self.creases
    }
    /// The tessellation node indices underlying each entry in [`creases`](Self::creases).
    pub fn crease_nodes(&self) -> &[[usize; 2]] {
        &self.crease_nodes
    }
    pub(super) fn of(tessellation: &Tessellation) -> Self {
        let coordinates = tessellation.mesh().coordinates();
        let triangles = triangles(tessellation);
        let sharp = crease_edges(&triangles, coordinates);
        let mut through = FxHashMap::<usize, Vec<usize>>::default();
        sharp.iter().for_each(|&[a, b]| {
            through.entry(a).or_default().push(b);
            through.entry(b).or_default().push(a)
        });
        let mut nodes: Vec<usize> = through.keys().copied().collect();
        nodes.sort_unstable();
        let corner_nodes: Vec<usize> = nodes
            .into_iter()
            .filter(|node| {
                let others = &through[node];
                match others.len() {
                    2 => {
                        let one = (&coordinates[others[0]] - &coordinates[*node]).normalized();
                        let two = (&coordinates[others[1]] - &coordinates[*node]).normalized();
                        &one * &two > -CREASE_COSINE
                    }
                    _ => true,
                }
            })
            .collect();
        let corners = corner_nodes
            .iter()
            .map(|&node| coordinates[node].clone())
            .collect();
        let creases = sharp
            .iter()
            .map(|&[a, b]| [coordinates[a].clone(), coordinates[b].clone()])
            .collect();
        Self {
            corners,
            corner_nodes,
            creases,
            crease_nodes: sharp,
        }
    }
    /// Bins the features so that everything within `radius` of a point is
    /// found in the twenty-seven cells about it.
    pub fn index(&self, radius: Quantity<Length>) -> FeatureIndex<'_> {
        let spacing = if radius > Quantity::new(0.0) {
            radius
        } else {
            Quantity::new(1.0)
        };
        let mut corners = FxHashMap::<[i64; D], Vec<usize>>::default();
        self.corners.iter().enumerate().for_each(|(index, point)| {
            corners.entry(cell(point, spacing)).or_default().push(index)
        });
        let mut creases = FxHashMap::<[i64; D], Vec<usize>>::default();
        let mut sprawling = Vec::new();
        self.creases.iter().enumerate().for_each(|(index, [a, b])| {
            let (low, high) = (cell(a, spacing), cell(b, spacing));
            let span: [i64; D] = from_fn(|axis| (high[axis] - low[axis]).abs() + 1);
            if span.iter().product::<i64>() > SPAN {
                return sprawling.push(index);
            }
            for i in low[0].min(high[0])..=low[0].max(high[0]) {
                for j in low[1].min(high[1])..=low[1].max(high[1]) {
                    for k in low[2].min(high[2])..=low[2].max(high[2]) {
                        creases.entry([i, j, k]).or_default().push(index)
                    }
                }
            }
        });
        FeatureIndex {
            features: self,
            corners,
            creases,
            sprawling,
            spacing,
        }
    }
    /// Labels each crease with the chain it belongs to: the maximal run of
    /// creases joined end to end at nodes that are not corners. Subdividing a
    /// crease adds only non-corner nodes, so the labeling partitions the same
    /// shape the same way however finely it is triangulated. `incident` maps
    /// each node to the creases through it.
    fn chains(&self, incident: &FxHashMap<usize, Vec<usize>>) -> Vec<usize> {
        fn root(parent: &mut [usize], mut crease: usize) -> usize {
            while parent[crease] != crease {
                parent[crease] = parent[parent[crease]];
                crease = parent[crease];
            }
            crease
        }
        let corner_nodes: FxHashSet<usize> = self.corner_nodes.iter().copied().collect();
        let mut parent: Vec<usize> = (0..self.creases.len()).collect();
        incident.iter().for_each(|(node, through)| {
            if !corner_nodes.contains(node) && through.len() == 2 {
                let (one, two) = (root(&mut parent, through[0]), root(&mut parent, through[1]));
                parent[one] = two;
            }
        });
        (0..self.creases.len())
            .map(|crease| root(&mut parent, crease))
            .collect()
    }
    /// Every *unrelated* crease within `radius` of each crease that surface
    /// runs between (see [`joined`]), nearest first, where a crease is related
    /// to every crease of its own chain and of every chain within `hops` steps
    /// of it (so a crease does not see its own smoothly-continuing neighbors
    /// as a narrow feature). Entries are empty when no such crease exists
    /// within `radius`.
    ///
    /// All of them are reported, not just the nearest, because a crease
    /// generally bounds a narrow feature along only part of its length: the
    /// creases nearest a long crease are often the short ones capping either
    /// end of the feature, which say nothing about the middle. Several
    /// creases are also routinely tied to the last bit for nearest, so
    /// choosing one of them would make the answer arbitrary.
    ///
    /// Chains, rather than individual creases, are what `hops` steps over:
    /// a chain is a maximal run of creases meeting end to end at nodes that
    /// are not corners, so the chain decomposition depends only on the shape
    /// and not on how finely the shape happens to be triangulated. Counting
    /// hops in creases instead would let a single sharp edge, once subdivided,
    /// pair its own collinear pieces with each other, reporting a "gap" of one
    /// triangle edge that shrinks with every refinement of the input.
    ///
    /// This is a proxy for local feature size that costs nothing beyond the
    /// crease detection already performed: two creases close in space but far
    /// apart along the surface (e.g. the two edges of a thin rib, wall, or
    /// stair-step) indicate a narrow region that a purely curvature- or
    /// thickness-driven size field would not otherwise catch, since the
    /// solid on either side of such a region need not be thin at all.
    ///
    /// The matched crease is reported alongside the distance because the
    /// narrow region itself is the space *between* the pair: a crease can be
    /// long while only one end of it runs close to anything, so the crease
    /// alone does not delimit where the feature is.
    pub fn separation(
        &self,
        tessellation: &Tessellation,
        radius: Quantity<Length>,
        hops: usize,
    ) -> Vec<Vec<Separation>> {
        if self.creases.is_empty() {
            return Vec::new();
        }
        let mut incident = FxHashMap::<usize, Vec<usize>>::default();
        self.crease_nodes
            .iter()
            .enumerate()
            .for_each(|(index, &[a, b])| {
                incident.entry(a).or_default().push(index);
                incident.entry(b).or_default().push(index);
            });
        let chain = self.chains(&incident);
        let mut adjacent = FxHashMap::<usize, FxHashSet<usize>>::default();
        incident.values().for_each(|through| {
            through.iter().for_each(|&one| {
                through.iter().for_each(|&two| {
                    if chain[one] != chain[two] {
                        adjacent.entry(chain[one]).or_default().insert(chain[two]);
                    }
                })
            })
        });
        // Which chains a chain may not be compared against depends only on
        // the chain, so the walk is done once per chain rather than per
        // crease.
        let mut excluded = FxHashMap::<usize, FxHashSet<usize>>::default();
        chain.iter().for_each(|&start| {
            excluded.entry(start).or_insert_with(|| {
                let mut reached = FxHashSet::<usize>::default();
                reached.insert(start);
                let mut frontier = vec![start];
                for _ in 0..hops {
                    let mut next = Vec::new();
                    frontier.iter().for_each(|near| {
                        if let Some(neighbors) = adjacent.get(near) {
                            neighbors.iter().for_each(|&other| {
                                if reached.insert(other) {
                                    next.push(other)
                                }
                            })
                        }
                    });
                    frontier = next;
                }
                reached
            });
        });
        // The creases of each chain running closest to each crease. Every
        // candidate pair is recorded from both ends, so the two creases agree
        // on it however the spatial index happened to turn one of them up,
        // and is measured once, from whichever end reached it first.
        let index = self.index(radius);
        let coordinates = tessellation.mesh().coordinates();
        let elements: Vec<&[usize]> = tessellation
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .collect();
        let bvh = tessellation.bvh();
        let mut nearest = vec![FxHashMap::<usize, Vec<Separation>>::default(); self.creases.len()];
        let mut measured = FxHashSet::<[usize; 2]>::default();
        let record = |nearest: &mut Vec<FxHashMap<usize, Vec<Separation>>>,
                      this: usize,
                      other: usize,
                      distance: Quantity<Length>| {
            nearest[this]
                .entry(chain[other])
                .or_default()
                .push(Separation {
                    crease: other,
                    distance,
                })
        };
        self.creases.iter().enumerate().for_each(|(this, segment)| {
            index
                .nearby_creases(&segment[0], &segment[1])
                .filter(|other| !excluded[&chain[this]].contains(&chain[*other]))
                .for_each(|other| {
                    if !measured.insert(key(this, other)) {
                        return;
                    }
                    let distance = segment_distance(segment, &self.creases[other]);
                    if distance < radius
                        && joined(segment, &self.creases[other], coordinates, &elements, bvh)
                    {
                        record(&mut nearest, this, other, distance);
                        record(&mut nearest, other, this, distance);
                    }
                })
        });
        // Every crease of a chain running as close to a crease as the closest
        // one of that chain does bounds the same stretch of the same narrow
        // feature, so all of them are kept. Keeping one per chain instead
        // would report a chain facing another along a subdivided stretch as a
        // single pair, leaving the rest of the stretch unbounded, and a
        // constant gap ties the crease directly opposite with the two
        // flanking it, which meet it at its own endpoints.
        (0..self.creases.len())
            .map(|this| {
                let mut near: Vec<Separation> = nearest[this]
                    .values()
                    .flat_map(|partners| {
                        let closest = partners
                            .iter()
                            .map(|partner| partner.distance)
                            .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
                            * SEPARATION_SPREAD;
                        partners
                            .iter()
                            .copied()
                            .filter(move |partner| partner.distance <= closest)
                    })
                    .collect();
                near.sort_by(|one, two| {
                    one.distance
                        .total_cmp(&two.distance)
                        .then(one.crease.cmp(&two.crease))
                });
                near
            })
            .collect()
    }
}

/// Whether surface runs between two creases the whole way along the stretch
/// they face each other over, so that the two of them bound one narrow ribbon
/// of surface rather than merely straddling open space.
///
/// A narrow feature is a narrow piece of *surface*: the thin face of a rib,
/// the tread of a stair-step, the floor of a slot. The two edges bounding
/// such a piece run close together with surface between them the whole way.
/// Two creases can also run just as close together with nothing between them
/// at all, though, which is what the mouth and the open top of a slot look
/// like: the two fins either side of it have sharp edges a gap apart along
/// their whole height, but the gap there is empty, and the model is no finer
/// there than the fins themselves are. Refining to the gap along those is
/// what would drive a whole slot, and not just its floor, down to the gap
/// size.
///
/// Each sample walks a point along one crease, drops to the nearest point of
/// the other, and asks whether the surface passes through the middle. A flat
/// ribbon passes exactly through it and a curved one misses by only its
/// sagitta, whereas across open space the nearest surface is one of the two
/// creases' own, half a gap away, so a quarter of the gap separates the two
/// cases with room to spare. Both creases are walked, since one may face only
/// part of the other.
fn joined(
    one: &[Coordinate<D>; 2],
    two: &[Coordinate<D>; 2],
    coordinates: &Coordinates<D>,
    elements: &[&[usize]],
    bvh: &BoundingVolumeHierarchy<D>,
) -> bool {
    [(one, two), (two, one)].into_iter().all(|(from, to)| {
        (0..JOINED_SAMPLES).all(|sample| {
            let fraction =
                Quantity::<Dimensionless>::new(sample as Scalar / (JOINED_SAMPLES - 1) as Scalar);
            let here = &from[0] + &(&(&from[1] - &from[0]) * fraction);
            let there = closest_on(to, &here);
            let gap = (&there - &here).norm();
            if gap == Quantity::default() {
                return true;
            }
            let middle =
                Coordinate::<D>::from(from_fn::<_, D, _>(|axis| (here[axis] + there[axis]) / 2.0));
            bvh.closest_point(&middle, coordinates, elements)
                .is_some_and(|(surface, _)| (&surface - &middle).norm() <= gap * JOINED_FRACTION)
        })
    })
}

/// Closest-point distance between two line segments (Ericson, *Real-Time
/// Collision Detection*, section 5.1.9).
fn segment_distance(one: &[Coordinate<D>; 2], two: &[Coordinate<D>; 2]) -> Quantity<Length> {
    let p1: [Scalar; D] = from_fn(|i| one[0][i].value());
    let q1: [Scalar; D] = from_fn(|i| one[1][i].value());
    let p2: [Scalar; D] = from_fn(|i| two[0][i].value());
    let q2: [Scalar; D] = from_fn(|i| two[1][i].value());
    let sub = |a: [Scalar; D], b: [Scalar; D]| -> [Scalar; D] { from_fn(|i| a[i] - b[i]) };
    let dot = |a: [Scalar; D], b: [Scalar; D]| -> Scalar { (0..D).map(|i| a[i] * b[i]).sum() };
    let d1 = sub(q1, p1);
    let d2 = sub(q2, p2);
    let r = sub(p1, p2);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    const EPSILON: Scalar = 1e-12;
    let (s, t) = if a <= EPSILON && e <= EPSILON {
        (0.0, 0.0)
    } else if a <= EPSILON {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= EPSILON {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denominator = a * e - b * b;
            let s = if denominator.abs() > EPSILON {
                ((b * f - c * e) / denominator).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let t = (b * s + f) / e;
            if t < 0.0 {
                ((-c / a).clamp(0.0, 1.0), 0.0)
            } else if t > 1.0 {
                (((b - c) / a).clamp(0.0, 1.0), 1.0)
            } else {
                (s, t)
            }
        }
    };
    let closest_one: [Scalar; D] = from_fn(|i| p1[i] + d1[i] * s);
    let closest_two: [Scalar; D] = from_fn(|i| p2[i] + d2[i] * t);
    let distance_squared: Scalar = (0..D)
        .map(|i| (closest_one[i] - closest_two[i]).powi(2))
        .sum();
    Quantity::new(distance_squared.sqrt())
}

fn cell(point: &Coordinate<D>, spacing: Quantity<Length>) -> [i64; D] {
    from_fn(|axis| (point[axis] / spacing).floor().value() as i64)
}

fn closest_on(segment: &[Coordinate<D>; 2], point: &Coordinate<D>) -> Coordinate<D> {
    let along = &segment[1] - &segment[0];
    let length = &along * &along;
    if length == Quantity::<Area>::new(0.0) {
        return segment[0].clone();
    }
    let fraction = Quantity::<Dimensionless>::new(
        ((point - &segment[0]) * &along / length)
            .value()
            .clamp(0.0, 1.0),
    );
    &segment[0] + &(along * fraction)
}

impl FeatureIndex<'_> {
    fn about(&self, point: &Coordinate<D>) -> Vec<[i64; D]> {
        let middle = cell(point, self.spacing);
        (-1..=1)
            .flat_map(|i| {
                (-1..=1).flat_map(move |j| {
                    (-1..=1).map(move |k| [middle[0] + i, middle[1] + j, middle[2] + k])
                })
            })
            .collect()
    }
    /// The corner nearest `point` within `radius`, and how far away it is.
    pub fn nearest_corner(
        &self,
        point: &Coordinate<D>,
        radius: Quantity<Length>,
    ) -> Option<(usize, Quantity<Length>)> {
        self.about(point)
            .iter()
            .filter_map(|cell| self.corners.get(cell))
            .flatten()
            .map(|&index| (index, (&self.features.corners[index] - point).norm()))
            .filter(|&(_, distance)| distance < radius)
            .min_by(|(_, one), (_, two)| one.total_cmp(two))
    }
    /// The point on a crease nearest `point` within `radius`.
    pub fn nearest_crease(
        &self,
        point: &Coordinate<D>,
        radius: Quantity<Length>,
    ) -> Option<Coordinate<D>> {
        self.about(point)
            .iter()
            .filter_map(|cell| self.creases.get(cell))
            .flatten()
            .chain(self.sprawling.iter())
            .map(|&index| closest_on(&self.features.creases[index], point))
            .map(|closest| {
                let distance = (&closest - point).norm();
                (closest, distance)
            })
            .filter(|&(_, distance)| distance < radius)
            .min_by(|(_, one), (_, two)| one.total_cmp(two))
            .map(|(closest, _)| closest)
    }
    /// The (deduplicated) crease indices whose bins overlap either endpoint
    /// of a segment, plus any sprawling creases, for the caller to filter and
    /// measure exactly against.
    fn nearby_creases(&self, a: &Coordinate<D>, b: &Coordinate<D>) -> impl Iterator<Item = usize> {
        let mut candidates: FxHashSet<usize> = self
            .about(a)
            .into_iter()
            .chain(self.about(b))
            .filter_map(|cell| self.creases.get(&cell))
            .flatten()
            .copied()
            .collect();
        candidates.extend(self.sprawling.iter().copied());
        candidates.into_iter()
    }
    pub fn corner(&self, index: usize) -> &Coordinate<D> {
        &self.features.corners[index]
    }
    pub fn is_empty(&self) -> bool {
        self.features.corners.is_empty() && self.features.creases.is_empty()
    }
}
