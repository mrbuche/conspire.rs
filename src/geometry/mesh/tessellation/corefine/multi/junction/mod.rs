//! The points where three surfaces meet.
//!
//! Where a triangle of each of three solids meets the other two in a segment, the three segments
//! are one point, where the three planes meet. It is not given, but defined by the corners of the
//! three triangles, and which side of a plane it is on is decided from those corners, so the answer
//! is exact however near the plane it is. The point is a junction when it is strictly inside all
//! three triangles, which the sides of the three planes through each edge of each triangle say.

#[cfg(test)]
mod test;

use super::{super::Feature, names::Groups};
use crate::geometry::{
    Coordinate,
    predicates::{Derived, orient3d, side},
};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, HashSet},
};

/// Where a point is on a triangle: inside it, on an edge, or on a corner. The edge `e` runs from
/// corner `e` to corner `e + 1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Located {
    Face,
    Edge(usize),
    Vertex(usize),
}

/// A point where one triangle of each of three solids meet, inside or on the boundary of all
/// three.
pub(super) struct Junction {
    /// The solids, in increasing order, and the triangle of each, numbered within it.
    pub triangles: [(usize, usize); 3],
    /// Where the point is on each of the triangles.
    pub located: [Located; 3],
    /// Where the point is, rounded: to be placed, never to be decided from.
    pub position: Coordinate<3>,
}

/// The junctions among the solids, given for each pair of solids the pairs of their triangles that
/// cross each other, and the corners of a triangle of a solid.
///
/// Fails where the three planes do not meet in one point.
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
                    if let Some((located, position)) = meeting(planes)? {
                        found.push(Junction {
                            triangles,
                            located,
                            position,
                        });
                    }
                }
            }
        }
    }
    Ok(found)
}

/// Where three triangles meet, and where that is on each, if it is inside or on all of them.
fn meeting(
    planes: [[&Coordinate<3>; 3]; 3],
) -> Result<Option<([Located; 3], Coordinate<3>)>, &'static str> {
    let point = Derived::Tpi(planes);
    let mut located = [Located::Face; 3];
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
        let mut on = Vec::new();
        for edge in 0..3 {
            let (u, v, t) = (
                triangle[edge],
                triangle[(edge + 1) % 3],
                triangle[(edge + 2) % 3],
            );
            let inside = orient3d(u, v, off, t);
            match side([u, v, off], &point)? {
                Ordering::Equal => on.push(edge),
                found if found != inside => return Ok(None),
                _ => (),
            }
        }
        located[index] = match on[..] {
            [] => Located::Face,
            [edge] => Located::Edge(edge),
            [0, 1] => Located::Vertex(1),
            [1, 2] => Located::Vertex(2),
            [0, 2] => Located::Vertex(0),
            _ => return Err("a point is on all three edges of a triangle"),
        };
    }
    Ok(Some((located, point.position()?)))
}

/// Where three or more surfaces meet, however many junctions of three of them are the one point.
pub(super) struct Point {
    /// The solid, the triangle of it, and where on that triangle the point is, for each of the
    /// surfaces.
    pub places: Vec<(usize, usize, Located)>,
    /// Where the point is, rounded: to be placed, never to be decided from.
    pub position: Coordinate<3>,
}

/// A curve along which two triangles cross, by the solids and their triangles.
pub(super) type Curve = ([usize; 2], usize, usize);

/// The ends of a curve: where they are, rounded, and the lowest feature of each of the two solids
/// that each is on.
pub(super) struct Ends {
    pub points: [Coordinate<3>; 2],
    pub features: [[Feature; 2]; 2],
}

/// The points on a curve: the one at its start and the one at its end, if any, and those between,
/// in order.
#[derive(Default)]
pub(super) struct Chain {
    pub start: Option<usize>,
    pub middle: Vec<usize>,
    pub end: Option<usize>,
}

type Arranged = (Vec<Point>, HashMap<Curve, Chain>);

/// The points, and the points on every curve that has any, where `curves` gives the ends of every
/// curve and `feature` the feature of a solid that a place on one of its triangles is.
///
/// A point is at the end of a curve when it is on the same features as the end is. Junctions that
/// are the same point, because a fourth surface meets the same three there, are one point. Fails
/// where points on a curve cannot be put in order, which is where one is too near an end of the
/// curve for its rounded position to say which side of the other surface it is on.
pub(super) fn arrange<'a>(
    junctions: &[Junction],
    curves: &HashMap<Curve, Ends>,
    feature: impl Fn(usize, usize, Located) -> Feature,
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
    let end_of = |curve: &Curve, x: usize| -> Option<usize> {
        let junction = &junctions[x];
        let mine = curve.0.map(|solid| {
            let n = junction
                .triangles
                .iter()
                .position(|&(s, _)| s == solid)
                .expect("a junction is on a curve of a solid it is not in");
            feature(solid, junction.triangles[n].1, junction.located[n])
        });
        (0..2).find(|&end| curves[curve].features[end] == mine)
    };
    let mut same = Groups::<usize>::default();
    for (curve, found) in &through {
        for (n, &x) in found.iter().enumerate() {
            for &y in &found[n + 1..] {
                let equal = match (end_of(curve, x), end_of(curve, y)) {
                    (Some(a), Some(b)) => a == b,
                    (None, None) => {
                        order(junctions, curve, &curves[curve].points, [x, y], corners)?
                            == Ordering::Equal
                    }
                    _ => false,
                };
                if equal {
                    same.join(&x, &y);
                }
            }
        }
    }
    let mut points = Vec::<Point>::new();
    let mut numbered = HashMap::<usize, usize>::new();
    for (index, junction) in junctions.iter().enumerate() {
        let root = same.find(&index);
        let next = points.len();
        let slot = *numbered.entry(root).or_insert(next);
        if slot == next {
            points.push(Point {
                places: Vec::new(),
                position: junctions[root].position.clone(),
            });
        }
        let point = &mut points[slot];
        for (&(solid, triangle), &located) in junction.triangles.iter().zip(&junction.located) {
            if !point.places.iter().any(|&(s, ..)| s == solid) {
                point.places.push((solid, triangle, located));
            }
        }
    }
    points
        .iter_mut()
        .for_each(|p| p.places.sort_unstable_by_key(|&(s, ..)| s));
    let mut arranged = HashMap::new();
    for (curve, found) in through {
        let mut chain = Chain::default();
        let mut roots = Vec::new();
        for &x in &found {
            let point = numbered[&same.find(&x)];
            match end_of(&curve, x) {
                Some(0) => chain.start = Some(point),
                Some(_) => chain.end = Some(point),
                None => roots.push(same.find(&x)),
            }
        }
        roots.sort_unstable();
        roots.dedup();
        for i in 1..roots.len() {
            let mut j = i;
            while j > 0 {
                let ends = &curves[&curve].points;
                match order(junctions, &curve, ends, [roots[j - 1], roots[j]], corners)? {
                    Ordering::Greater => roots.swap(j - 1, j),
                    Ordering::Equal => return Err("two triple points are the same point"),
                    Ordering::Less => break,
                }
                j -= 1;
            }
        }
        chain.middle = roots.iter().map(|root| numbered[root]).collect();
        arranged.insert(curve, chain);
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
