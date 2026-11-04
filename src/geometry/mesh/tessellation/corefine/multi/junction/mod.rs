//! The points where three surfaces meet.
//!
//! Where a triangle of each of three solids meets the other two in a segment, the three segments
//! are one point, where the three planes meet. It is not given, but defined by the corners of the
//! three triangles, and which side of a plane it is on is decided from those corners, so the answer
//! is exact however near the plane it is. The point is a junction when it is strictly inside all
//! three triangles, which the sides of the three planes through each edge of each triangle say.

#[cfg(test)]
mod test;

use super::names::Groups;
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

/// Where three or more surfaces meet, however many junctions of three of them are the one point.
pub(super) struct Point {
    /// The solid and the triangle of it that the point is inside, for each of the surfaces.
    pub triangles: Vec<(usize, usize)>,
    /// Where the point is, rounded: to be placed, never to be decided from.
    pub position: Coordinate<3>,
}

/// A curve along which two triangles cross, by the solids and their triangles.
pub(super) type Curve = ([usize; 2], usize, usize);

/// The points, and the indices of those on each curve that has any, from its start to its end.
type Arranged = (Vec<Point>, HashMap<Curve, Vec<usize>>);

/// The points, and for each curve that has any the indices of those on it from its start to its
/// end, where `ends` gives the ends of every curve.
///
/// Junctions that are the same point, because a fourth surface meets the same three there, are one
/// point. Fails where points on a curve cannot be put in order, which is where one is too near an
/// end of the curve for its rounded position to say which side of the other surface it is on.
pub(super) fn arrange<'a>(
    junctions: &[Junction],
    ends: &HashMap<Curve, [Coordinate<3>; 2]>,
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3] + Copy,
) -> Result<Arranged, &'static str> {
    let mut through = HashMap::<Curve, Vec<usize>>::new();
    for (index, junction) in junctions.iter().enumerate() {
        for (n, &(first, a)) in junction.triangles.iter().enumerate() {
            for &(second, b) in &junction.triangles[n + 1..] {
                through
                    .entry(([first, second], a, b))
                    .or_default()
                    .push(index);
            }
        }
    }
    let mut same = Groups::<usize>::default();
    for (curve, found) in &through {
        for (n, &x) in found.iter().enumerate() {
            for &y in &found[n + 1..] {
                if order(junctions, curve, &ends[curve], [x, y], corners)? == Ordering::Equal {
                    same.join(&x, &y);
                }
            }
        }
    }
    let mut points = Vec::new();
    let mut numbered = HashMap::<usize, usize>::new();
    for index in 0..junctions.len() {
        let root = same.find(&index);
        let next = points.len();
        let slot = *numbered.entry(root).or_insert(next);
        if slot == next {
            points.push(Point {
                triangles: Vec::new(),
                position: junctions[root].position.clone(),
            });
        }
        let point = &mut points[slot];
        for &triangle in &junctions[index].triangles {
            if !point.triangles.contains(&triangle) {
                point.triangles.push(triangle);
            }
        }
    }
    points.iter_mut().for_each(|p| p.triangles.sort_unstable());
    let mut arranged = HashMap::new();
    for (curve, found) in through {
        let mut roots: Vec<usize> = found.iter().map(|index| same.find(index)).collect();
        roots.sort_unstable();
        roots.dedup();
        for i in 1..roots.len() {
            let mut j = i;
            while j > 0 {
                let ends = &ends[&curve];
                match order(junctions, &curve, ends, [roots[j - 1], roots[j]], corners)? {
                    Ordering::Greater => roots.swap(j - 1, j),
                    Ordering::Equal => return Err("two triple points are the same point"),
                    Ordering::Less => break,
                }
                j -= 1;
            }
        }
        arranged.insert(curve, roots.iter().map(|root| numbered[root]).collect());
    }
    Ok((points, arranged))
}

/// Whether the junction `x` is before, after or the same point as `y` along a curve from its start
/// to its end.
///
/// The point `x` is on the plane of the third triangle that meets there. The side of that plane
/// that `y` is on is the side of it that the curve runs to after `x`, which is the side of its end,
/// when it is after `x`.
fn order<'a>(
    junctions: &[Junction],
    curve: &Curve,
    ends: &[Coordinate<3>; 2],
    [x, y]: [usize; 2],
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
) -> Result<Ordering, &'static str> {
    let (pair, ..) = curve;
    let &(solid, triangle) = junctions[x]
        .triangles
        .iter()
        .find(|(solid, _)| !pair.contains(solid))
        .ok_or("a triple point has no third surface")?;
    let plane = corners(solid, triangle);
    let planes = junctions[y]
        .triangles
        .map(|(solid, triangle)| corners(solid, triangle));
    let found = side(plane, &Derived::Tpi(planes))?;
    if found == Ordering::Equal {
        return Ok(Ordering::Equal);
    }
    let [before, after] = ends
        .each_ref()
        .map(|end| orient3d(plane[0], plane[1], plane[2], end));
    if before == after || before == Ordering::Equal || after == Ordering::Equal {
        return Err("a triple point is too near the end of a curve to be put in order");
    }
    Ok(if found == after {
        Ordering::Less
    } else {
        Ordering::Greater
    })
}
