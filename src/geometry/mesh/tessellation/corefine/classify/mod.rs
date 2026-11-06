//! Where the pieces of two split tessellations lie relative to each other's solid.

#[cfg(test)]
mod test;

use super::{
    Contacts, Feature, Key, Side,
    split::{Split, split},
};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Tessellation, Watertight},
        predicates::orient2d,
        solid::SolidOracle,
    },
    math::{CrossProduct, FxHashMap, FxHashSet, Quantity, Reference, Scalar, Tensor, TensorRank1},
    units::{Area, Length},
};
use alloc::collections::BTreeSet;
use core::cmp::Ordering;

/// How far a point can be from a surface, as a fraction of `reach`, before it cannot be told to be
/// off it. It is a few thousand times the rounding of the coordinates it is found from.
const INDISTINGUISHABLE: Scalar = 1e-12;

/// Where a piece of the surface of one solid lies relative to another solid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Region {
    /// Outside the other solid.
    Outside,
    /// Inside the other solid.
    Inside,
    /// On the surface of the other solid, facing the same way if `aligned` and opposite otherwise.
    Coincident { aligned: bool },
}

/// Two solids with their surfaces split along each other, and where each piece of each surface
/// lies relative to the other solid.
pub struct Corefined {
    pub first: Tessellation,
    pub second: Tessellation,
    /// The region of each triangle of `first`, relative to the second solid.
    pub first_regions: Vec<Region>,
    /// The region of each triangle of `second`, relative to the first solid.
    pub second_regions: Vec<Region>,
    /// The vertices of `first` and `second` that are the same point, as pairs, named by the keys
    /// of where the two surfaces meet.
    pub(super) shared: Vec<[usize; 2]>,
}

impl Watertight {
    /// This solid and the other with their surfaces split along each other, and every triangle
    /// of each classified against the other solid.
    ///
    /// The triangles are grouped into patches, which are connected across every edge that is not
    /// where the other surface meets this one, and each patch is classified once, by its largest
    /// triangle. A patch that lies in a plane with a triangle of the other surface is
    /// [`Region::Coincident`]. Any other is on one side of the other surface, and fails if it is
    /// too close to tell.
    pub fn corefine(&self, other: &Watertight) -> Result<Corefined, &'static str> {
        let (first, second) = (self.tessellation(), other.tessellation());
        let contacts = first.contacts(second)?;
        let (a, b) = (
            split(first, second, Side::First, &contacts)?,
            split(second, first, Side::Second, &contacts)?,
        );
        let keys: BTreeSet<&Key> = contacts
            .crossings
            .iter()
            .flat_map(|crossing| &crossing.keys)
            .chain(contacts.constraints.iter().flat_map(|c| &c.keys))
            .collect();
        let vertex = |feature: &Feature, split: &Split, key: &Key| match feature {
            Feature::Vertex(node) => Some(*node),
            _ => split.named.get(key).copied(),
        };
        let shared = keys
            .into_iter()
            .filter_map(|key| Some([vertex(&key.first, &a, key)?, vertex(&key.second, &b, key)?]))
            .collect();
        Ok(Corefined {
            first_regions: regions(&a, second, other, &coplanar(&contacts, Side::First))?,
            second_regions: regions(&b, first, self, &coplanar(&contacts, Side::Second))?,
            first: a.tessellation,
            second: b.tessellation,
            shared,
        })
    }
}

/// For each triangle of one tessellation, the triangles of the other in its plane.
pub(super) fn coplanar(contacts: &Contacts, side: Side) -> FxHashMap<usize, Vec<usize>> {
    let mut found = FxHashMap::<usize, Vec<usize>>::default();
    contacts.coplanar.iter().for_each(|&[first, second]| {
        let (own, other) = match side {
            Side::First => (first, second),
            Side::Second => (second, first),
        };
        found.entry(own).or_default().push(other)
    });
    found
}

fn regions(
    split: &Split,
    other: &Tessellation,
    solid: &Watertight,
    coplanar: &FxHashMap<usize, Vec<usize>>,
) -> Result<Vec<Region>, &'static str> {
    let coordinates = split.tessellation.mesh().coordinates();
    let faces: Vec<[usize; 3]> = split
        .tessellation
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|nodes| [nodes[0], nodes[1], nodes[2]])
        .collect();
    relate(
        &faces,
        |node| &coordinates[node],
        &split.parents,
        &patches(&faces, &split.curves),
        other,
        solid,
        coplanar,
    )
}

/// The groups of triangles that are connected across every edge that is not one of the `curves`.
pub(super) fn patches(faces: &[[usize; 3]], curves: &FxHashSet<[usize; 2]>) -> Vec<Vec<usize>> {
    let mut around = FxHashMap::<[usize; 2], Vec<usize>>::default();
    for (index, nodes) in faces.iter().enumerate() {
        (0..3).for_each(|edge| {
            let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        })
    }
    let mut found = Vec::new();
    let mut seen = vec![false; faces.len()];
    for start in 0..faces.len() {
        if seen[start] {
            continue;
        }
        let mut patch = vec![start];
        seen[start] = true;
        let mut next = 0;
        while next < patch.len() {
            let nodes = faces[patch[next]];
            next += 1;
            for edge in 0..3 {
                let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
                let key = [a.min(b), a.max(b)];
                if curves.contains(&key) {
                    continue;
                }
                for &neighbour in &around[&key] {
                    if !seen[neighbour] {
                        seen[neighbour] = true;
                        patch.push(neighbour);
                    }
                }
            }
        }
        found.push(patch);
    }
    found
}

/// Where each triangle of a surface lies relative to the `solid` of the `other` surface, with the
/// `patches` of triangles that cannot differ classified once, by the largest triangle of each.
///
/// The triangles name their corners by `faces`, which `position` places, and `parents` gives the
/// triangle of the surface that each was split from, for the triangles of `other` in its plane.
pub(super) fn relate<'c>(
    faces: &[[usize; 3]],
    position: impl Fn(usize) -> &'c Coordinate<3>,
    parents: &[usize],
    patches: &[Vec<usize>],
    other: &Tessellation,
    solid: &Watertight,
    coplanar: &FxHashMap<usize, Vec<usize>>,
) -> Result<Vec<Region>, &'static str> {
    let corners = |face: usize| faces[face].map(&position);
    let other_coordinates = other.mesh().coordinates();
    let other_faces: Vec<&[usize]> = other.mesh().connectivities().iter().flatten().collect();
    let mut regions: Vec<Option<Region>> = vec![None; faces.len()];
    for patch in patches {
        let largest = *patch
            .iter()
            .max_by(|&&a, &&b| {
                let size = |face| normal(corners(face)).norm().value();
                size(a).total_cmp(&size(b))
            })
            .ok_or("a patch has no triangle")?;
        let triangle = corners(largest);
        let centre = triangle.into_iter().sum::<Coordinate<3>>() / 3.0;
        let direction = normal(triangle).normalized();
        let on = coplanar
            .get(&parents[largest])
            .into_iter()
            .flatten()
            .find_map(|&index| {
                let nodes = other_faces[index];
                let corners = [nodes[0], nodes[1], nodes[2]].map(|node| &other_coordinates[node]);
                inside(&centre, corners).then(|| Region::Coincident {
                    aligned: (&direction * normal(corners).normalized()).value() > 0.0,
                })
            });
        let region = match on {
            Some(region) => region,
            None => {
                let distance = solid.signed_distance(&centre);
                if distance.abs() < reach(&centre, triangle) * INDISTINGUISHABLE {
                    return Err("a piece lies on the other surface but not in its triangles");
                }
                if distance < Length::meters(0.0) {
                    Region::Inside
                } else {
                    Region::Outside
                }
            }
        };
        patch.iter().for_each(|&face| regions[face] = Some(region));
    }
    Ok(regions.into_iter().flatten().collect())
}

/// Whether a point in the plane of a triangle is strictly inside it.
fn inside(point: &Coordinate<3>, triangle: [&Coordinate<3>; 3]) -> bool {
    let n = normal(triangle);
    let axis = (0..3)
        .max_by(|&a, &b| n[a].value().abs().total_cmp(&n[b].value().abs()))
        .unwrap_or(2);
    let signs =
        [0, 1, 2].map(|edge| orient2d(triangle[edge], triangle[(edge + 1) % 3], point, axis));
    signs[0] != Ordering::Equal && signs[0] == signs[1] && signs[1] == signs[2]
}

/// The length rounding is measured against for the centre of a triangle: how far it is from the
/// origin, which sets the spacing of the numbers its coordinates are held in, or how big the
/// triangle is, whichever is more.
fn reach(centre: &Coordinate<3>, [a, b, c]: [&Coordinate<3>; 3]) -> Quantity<Length> {
    [
        (b - a).norm(),
        (c - b).norm(),
        (a - c).norm(),
        centre.norm(),
    ]
    .into_iter()
    .fold(Quantity::default(), Quantity::max)
}

/// Twice the area of a triangle, in the direction its corners turn.
fn normal([a, b, c]: [&Coordinate<3>; 3]) -> TensorRank1<3, Reference, Area> {
    (&(b - a)).cross(&(c - a))
}
