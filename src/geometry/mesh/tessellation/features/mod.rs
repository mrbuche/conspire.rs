#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Connectivity,
            tessellation::{D, Tessellation},
        },
    },
    math::{CrossProduct, FxHashMap, Scalar, Tensor},
};
use std::{array::from_fn, f64::consts::FRAC_1_SQRT_2};

const CREASE_COSINE: Scalar = FRAC_1_SQRT_2;

/// The sharp edges and points of a tessellation.
///
/// An edge is a crease when its two triangles turn through more than
/// forty-five degrees. A vertex is a corner when the creases through it do
/// not simply pass through: it ends one, joins three or more, or turns
/// sharply between two.
///
/// Only edges with exactly two incident triangles are considered, so a
/// tessellation whose triangles do not share nodes has no features rather
/// than having every edge be one.
pub struct Features {
    corners: Vec<Coordinate<D>>,
    creases: Vec<[Coordinate<D>; 2]>,
    /// Which corner, if any, each end of a crease is.
    ends: Vec<[Option<usize>; 2]>,
}

/// The features of a tessellation, binned for lookup within a fixed radius.
pub struct FeatureIndex<'a> {
    features: &'a Features,
    corners: FxHashMap<[i64; D], Vec<usize>>,
    creases: FxHashMap<[i64; D], Vec<usize>>,
    /// Creases spanning too many cells to bin, scanned by every lookup.
    sprawling: Vec<usize>,
    spacing: Scalar,
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

impl Features {
    pub fn corners(&self) -> &[Coordinate<D>] {
        &self.corners
    }
    pub fn creases(&self) -> &[[Coordinate<D>; 2]] {
        &self.creases
    }
    pub(super) fn of(tessellation: &Tessellation) -> Self {
        let coordinates = tessellation.mesh().coordinates();
        let triangles = triangles(tessellation);
        let normals: Vec<Coordinate<D>> = triangles
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
            .filter(|(_, triangles)| {
                &normals[triangles[0]] * &normals[triangles[1]] < CREASE_COSINE
            })
            .map(|(&edge, _)| edge)
            .collect();
        sharp.sort_unstable();
        let mut through = FxHashMap::<usize, Vec<usize>>::default();
        sharp.iter().for_each(|&[a, b]| {
            through.entry(a).or_default().push(b);
            through.entry(b).or_default().push(a)
        });
        let mut nodes: Vec<usize> = through.keys().copied().collect();
        nodes.sort_unstable();
        let cornered: Vec<usize> = nodes
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
        let at: FxHashMap<usize, usize> = cornered
            .iter()
            .enumerate()
            .map(|(index, &node)| (node, index))
            .collect();
        let ends = sharp
            .iter()
            .map(|[a, b]| [at.get(a).copied(), at.get(b).copied()])
            .collect();
        let corners = cornered
            .into_iter()
            .map(|node| coordinates[node].clone())
            .collect();
        let creases = sharp
            .into_iter()
            .map(|[a, b]| [coordinates[a].clone(), coordinates[b].clone()])
            .collect();
        Self {
            corners,
            creases,
            ends,
        }
    }
    /// The corner every one of the given creases ends at, if they share one.
    pub fn hub(&self, creases: &[usize]) -> Option<usize> {
        let (first, rest) = creases.split_first()?;
        self.ends[*first].into_iter().flatten().find(|corner| {
            rest.iter()
                .all(|&crease| self.ends[crease].contains(&Some(*corner)))
        })
    }
    /// Bins the features so that everything within `radius` of a point is
    /// found in the twenty-seven cells about it.
    pub fn index(&self, radius: Scalar) -> FeatureIndex<'_> {
        let spacing = if radius > 0.0 { radius } else { 1.0 };
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
}

fn cell(point: &Coordinate<D>, spacing: Scalar) -> [i64; D] {
    from_fn(|axis| (point[axis] / spacing).floor() as i64)
}

/// Where a segment meets a triangle, as a fraction along the segment.
fn meets(segment: &[Coordinate<D>; 2], triangle: [&Coordinate<D>; 3]) -> Option<Scalar> {
    let along = &segment[1] - &segment[0];
    let (one, two) = (triangle[1] - triangle[0], triangle[2] - triangle[0]);
    let normal = along.cross(&two);
    let determinant = &one * &normal;
    let scale = one.norm() * two.norm() * along.norm();
    if determinant.abs() < CROSSING * scale {
        return None;
    }
    let inverse = 1.0 / determinant;
    let offset = &segment[0] - triangle[0];
    let u = inverse * (&offset * &normal);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let across = offset.cross(&one);
    let v = inverse * (&along * &across);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let fraction = inverse * (&two * &across);
    (0.0..=1.0).contains(&fraction).then_some(fraction)
}

/// Relative tolerance below which a crease is taken to run along a face
/// rather than through it.
const CROSSING: Scalar = 1.0e-12;

fn closest_on(segment: &[Coordinate<D>; 2], point: &Coordinate<D>) -> Coordinate<D> {
    let along = &segment[1] - &segment[0];
    let length = &along * &along;
    if length == 0.0 {
        return segment[0].clone();
    }
    let fraction = ((point - &segment[0]) * &along / length).clamp(0.0, 1.0);
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
    pub fn nearest_corner(&self, point: &Coordinate<D>, radius: Scalar) -> Option<(usize, Scalar)> {
        self.about(point)
            .iter()
            .filter_map(|cell| self.corners.get(cell))
            .flatten()
            .map(|&index| (index, (&self.features.corners[index] - point).norm()))
            .filter(|&(_, distance)| distance < radius)
            .min_by(|(_, one), (_, two)| one.total_cmp(two))
    }
    /// The point on a crease nearest `point` within `radius`.
    pub fn nearest_crease(&self, point: &Coordinate<D>, radius: Scalar) -> Option<Coordinate<D>> {
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
    pub fn corner(&self, index: usize) -> &Coordinate<D> {
        &self.features.corners[index]
    }
    /// Which creases pass through the quad, and where.
    ///
    /// The quad is taken as two triangles about its first corner, so a
    /// background face that is not planar is still covered. A crease meeting
    /// the shared diagonal is reported once.
    pub fn through(&self, quad: [&Coordinate<D>; 4]) -> Vec<(usize, Coordinate<D>)> {
        let mut through: Vec<(usize, Coordinate<D>)> = self
            .features
            .creases
            .iter()
            .enumerate()
            .filter_map(|(index, crease)| {
                [[quad[0], quad[1], quad[2]], [quad[0], quad[2], quad[3]]]
                    .into_iter()
                    .find_map(|triangle| meets(crease, triangle))
                    .map(|fraction| {
                        let point = &crease[0] + &((&crease[1] - &crease[0]) * fraction);
                        (index, point)
                    })
            })
            .collect();
        through.sort_by_key(|&(index, _)| index);
        through
    }
    pub fn is_empty(&self) -> bool {
        self.features.corners.is_empty() && self.features.creases.is_empty()
    }
}
