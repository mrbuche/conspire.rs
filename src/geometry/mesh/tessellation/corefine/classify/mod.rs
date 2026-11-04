//! Where the pieces of two split tessellations lie relative to each other's solid.

#[cfg(test)]
mod test;

use super::{
    Contacts, Side,
    split::{Split, split},
};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Tessellation, Watertight},
        predicates::orient2d,
        solid::SolidOracle,
    },
    units::Length,
};
use std::{array::from_fn, cmp::Ordering, collections::HashMap};

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
        Ok(Corefined {
            first_regions: regions(&a, second, other, &coplanar(&contacts, Side::First))?,
            second_regions: regions(&b, first, self, &coplanar(&contacts, Side::Second))?,
            first: a.tessellation,
            second: b.tessellation,
        })
    }
}

/// For each triangle of one tessellation, the triangles of the other in its plane.
fn coplanar(contacts: &Contacts, side: Side) -> HashMap<usize, Vec<usize>> {
    let mut found = HashMap::<usize, Vec<usize>>::new();
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
    coplanar: &HashMap<usize, Vec<usize>>,
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
    let corners = |face: usize| faces[face].map(|node| values(&coordinates[node]));
    let other_coordinates = other.mesh().coordinates();
    let other_faces: Vec<&[usize]> = other.mesh().connectivities().iter().flatten().collect();
    let mut around = HashMap::<[usize; 2], Vec<usize>>::new();
    for (index, nodes) in faces.iter().enumerate() {
        (0..3).for_each(|edge| {
            let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        })
    }
    let mut regions: Vec<Option<Region>> = vec![None; faces.len()];
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
                if split.curves.contains(&key) {
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
        let largest = *patch
            .iter()
            .max_by(|&&a, &&b| norm(normal(corners(a))).total_cmp(&norm(normal(corners(b)))))
            .unwrap_or(&start);
        let triangle = corners(largest);
        let centre: [f64; 3] =
            from_fn(|axis| triangle.iter().map(|corner| corner[axis]).sum::<f64>() / 3.0);
        let direction = normal(triangle);
        let on = coplanar
            .get(&split.parents[largest])
            .into_iter()
            .flatten()
            .find_map(|&index| {
                let nodes = other_faces[index];
                let corners = [nodes[0], nodes[1], nodes[2]].map(|node| &other_coordinates[node]);
                inside(centre, corners).then(|| Region::Coincident {
                    aligned: dot(direction, normal(corners.map(values))) > 0.0,
                })
            });
        let region = match on {
            Some(region) => region,
            None => {
                let distance = solid.signed_distance(&centre.into());
                if distance.abs() < Length::meters(1e-12) {
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
fn inside(point: [f64; 3], triangle: [&Coordinate<3>; 3]) -> bool {
    let n = normal(triangle.map(values));
    let axis = (0..3)
        .max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
        .unwrap_or(2);
    let point: Coordinate<3> = point.into();
    let signs =
        [0, 1, 2].map(|edge| orient2d(triangle[edge], triangle[(edge + 1) % 3], &point, axis));
    signs[0] != Ordering::Equal && signs[0] == signs[1] && signs[1] == signs[2]
}

fn values(point: &Coordinate<3>) -> [f64; 3] {
    [0, 1, 2].map(|axis| point[axis].value())
}

fn normal(triangle: [[f64; 3]; 3]) -> [f64; 3] {
    let e = |i: usize| -> [f64; 3] { from_fn(|k| triangle[i][k] - triangle[0][k]) };
    let (a, b) = (e(1), e(2));
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| a[k] * b[k]).sum()
}

fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
