//! The points where three surfaces meet.
//!
//! Where a triangle of each of three solids meets the other two in a segment, the three segments
//! are one point, where the three planes meet. It is not given, but defined by the corners of the
//! three triangles, and which side of a plane it is on is decided from those corners, so the answer
//! is exact however near the plane it is. The point is a junction when it is strictly inside all
//! three triangles, which the sides of the three planes through each edge of each triangle say.

#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinate,
    predicates::{Derived, orient3d, side},
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
};

/// A point where one triangle of each of three solids meet, strictly inside all three.
pub(super) struct Junction {
    /// The solids, in increasing order, and the triangle of each, numbered within it.
    pub triangles: [(usize, usize); 3],
    /// Where the point is, rounded: to be placed, never to be decided from.
    pub position: Coordinate<3>,
}

/// The junctions among the solids, given for each pair of solids the pairs of their triangles that
/// cross each other, and the corners of a triangle of a solid.
///
/// Fails where the three planes do not meet in one point, or the point is on the boundary of a
/// triangle, which are not yet cut at.
pub(super) fn junctions<'a>(
    pairs: &BTreeMap<[usize; 2], HashSet<[usize; 2]>>,
    solids: usize,
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
) -> Result<Vec<Junction>, &'static str> {
    let mut found = Vec::new();
    for i in 0..solids {
        for j in i + 1..solids {
            for k in j + 1..solids {
                let (Some(ij), Some(ik), Some(jk)) =
                    (pairs.get(&[i, j]), pairs.get(&[i, k]), pairs.get(&[j, k]))
                else {
                    continue;
                };
                let mut with_first = HashMap::<usize, Vec<usize>>::new();
                ik.iter()
                    .for_each(|&[a, c]| with_first.entry(a).or_default().push(c));
                let mut candidates: Vec<[usize; 3]> = ij
                    .iter()
                    .flat_map(|&[a, b]| {
                        with_first
                            .get(&a)
                            .into_iter()
                            .flatten()
                            .map(move |&c| [a, b, c])
                    })
                    .filter(|&[_, b, c]| jk.contains(&[b, c]))
                    .collect();
                candidates.sort_unstable();
                for [a, b, c] in candidates {
                    let triangles = [(i, a), (j, b), (k, c)];
                    let planes = triangles.map(|(solid, triangle)| corners(solid, triangle));
                    if let Some(position) = meeting(planes)? {
                        found.push(Junction {
                            triangles,
                            position,
                        });
                    }
                }
            }
        }
    }
    Ok(found)
}

/// Where three triangles meet, if it is strictly inside all of them.
fn meeting(planes: [[&Coordinate<3>; 3]; 3]) -> Result<Option<Coordinate<3>>, &'static str> {
    let point = Derived::Tpi(planes);
    let mut on_boundary = false;
    for (index, triangle) in planes.iter().enumerate() {
        let others = [planes[(index + 1) % 3], planes[(index + 2) % 3]];
        let off = others
            .iter()
            .flatten()
            .find(|&&corner| {
                orient3d(triangle[0], triangle[1], triangle[2], corner) != Ordering::Equal
            })
            .ok_or("two of three surfaces that meet are in one plane")?;
        let off = *off;
        for edge in 0..3 {
            let (u, v, t) = (
                triangle[edge],
                triangle[(edge + 1) % 3],
                triangle[(edge + 2) % 3],
            );
            let inside = orient3d(u, v, off, t);
            match side([u, v, off], &point)? {
                Ordering::Equal => on_boundary = true,
                found if found != inside => return Ok(None),
                _ => (),
            }
        }
    }
    if on_boundary {
        return Err("three surfaces meet on the edge of a triangle");
    }
    Ok(Some(point.position()?))
}
