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
use crate::math::{FxHashMap, FxHashSet};
use alloc::collections::BTreeMap;
use core::cmp::Ordering;

/// Where a point is on a triangle: inside it, on an edge, or on a corner. The edge `e` runs from
/// corner `e` to corner `e + 1`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Located {
    Face,
    Edge(usize),
    Vertex(usize),
}

/// How a junction is defined by the triangles that meet there.
pub(super) enum How {
    /// By the three planes of the triangles.
    Planes,
    /// By an edge of a triangle, where it meets the plane of another. The triangle is one of two in
    /// one plane, so the three planes do not meet in a point.
    Edge {
        solid: usize,
        triangle: usize,
        edge: usize,
        plane: (usize, usize),
    },
    /// By an edge of one triangle and an edge of another, in the plane that three triangles share,
    /// where the edges cross. The plane through the second edge and a corner `off` of some
    /// triangle that is not in the shared plane, cuts the first edge at the point.
    Edges {
        first: (usize, usize, usize),
        second: (usize, usize, usize),
        off: (usize, usize, usize),
    },
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
    pub how: How,
}

impl Junction {
    /// The point, as defined by the corners of the triangles.
    fn derived<'a>(&self, corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3]) -> Derived<'a> {
        match &self.how {
            How::Planes => Derived::Tpi(
                self.triangles
                    .map(|(solid, triangle)| corners(solid, triangle)),
            ),
            How::Edge {
                solid,
                triangle,
                edge,
                plane,
            } => {
                let owner = corners(*solid, *triangle);
                Derived::Lpi {
                    line: [owner[*edge], owner[(*edge + 1) % 3]],
                    plane: corners(plane.0, plane.1),
                }
            }
            How::Edges { first, second, off } => {
                let (x, y) = (corners(first.0, first.1), corners(second.0, second.1));
                let off = corners(off.0, off.1)[off.2];
                Derived::Lpi {
                    line: [x[first.2], x[(first.2 + 1) % 3]],
                    plane: [y[second.2], y[(second.2 + 1) % 3], off],
                }
            }
        }
    }
    /// Planes through the point, each through three given points, one of which cuts across the
    /// curve along which two of the surfaces cross unless the point is on all of them.
    fn across<'a>(
        &self,
        pair: &[usize; 2],
        corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
    ) -> Result<Vec<[&'a Coordinate<3>; 3]>, &'static str> {
        match &self.how {
            How::Planes => self
                .triangles
                .iter()
                .find(|(solid, _)| !pair.contains(solid))
                .map(|&(solid, triangle)| vec![corners(solid, triangle)])
                .ok_or("a triple point has no third surface"),
            How::Edge {
                solid,
                triangle,
                edge,
                plane,
            } => {
                let owner = corners(*solid, *triangle);
                let off = corners(plane.0, plane.1)
                    .into_iter()
                    .find(|corner| {
                        orient3d(owner[0], owner[1], owner[2], corner) != Ordering::Equal
                    })
                    .ok_or("a triangle lies in the plane of another")?;
                Ok(vec![
                    [owner[*edge], owner[(*edge + 1) % 3], off],
                    corners(plane.0, plane.1),
                ])
            }
            How::Edges { .. } => Err("a point in a plane shared by three surfaces is on no curve"),
        }
    }
}

/// The junctions among the solids, given for each pair of solids the pairs of their triangles that
/// cross each other, and the corners of a triangle of a solid.
///
/// Fails where the three planes do not meet in one point.
pub(super) fn junctions<'a>(
    pairs: &BTreeMap<[usize; 2], FxHashSet<[usize; 2]>>,
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
                let mut with_first = FxHashMap::<usize, Vec<usize>>::default();
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
                            how: How::Planes,
                        });
                    }
                }
            }
        }
    }
    Ok(found)
}

/// The junctions where an edge of one of three triangles in one plane crosses an edge of another,
/// inside or on the third, given for each pair of solids the pairs of their triangles that lie in
/// one plane, and how many triangles each solid has.
///
/// The three planes are one, so the point is where two edges cross in it. Where it is is decided
/// with the plane through one edge and a corner of some triangle that is not in the plane.
pub(super) fn planar_junctions<'a>(
    coplanar: &BTreeMap<[usize; 2], Vec<[usize; 2]>>,
    counts: &[usize],
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
) -> Result<Vec<Junction>, &'static str> {
    let mut partners = FxHashMap::<(usize, usize, usize), Vec<usize>>::default();
    for (&[x, y], list) in coplanar {
        for &[tx, ty] in list {
            partners.entry((y, x, ty)).or_default().push(tx);
            partners.entry((x, y, tx)).or_default().push(ty);
        }
    }
    let mut found = Vec::new();
    for (&[i, j], list) in coplanar {
        for &[a, b] in list {
            for k in (j + 1)..counts.len() {
                let on_a: FxHashSet<usize> = partners
                    .get(&(i, k, a))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                let on_b: Vec<usize> = partners
                    .get(&(j, k, b))
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|c| on_a.contains(c))
                    .collect();
                for c in on_b {
                    found.extend(planar_triple([(i, a), (j, b), (k, c)], counts, &corners)?);
                }
            }
        }
    }
    Ok(found)
}

/// The junctions where two edges of three triangles in one plane cross.
fn planar_triple<'a>(
    triangles: [(usize, usize); 3],
    counts: &[usize],
    corners: &impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
) -> Result<Vec<Junction>, &'static str> {
    let shape = triangles.map(|(solid, triangle)| corners(solid, triangle));
    let mut off = None;
    'search: for (solid, &count) in counts.iter().enumerate() {
        for triangle in 0..count {
            for (corner, point) in corners(solid, triangle).into_iter().enumerate() {
                if orient3d(shape[0][0], shape[0][1], shape[0][2], point) != Ordering::Equal {
                    off = Some((solid, triangle, corner));
                    break 'search;
                }
            }
        }
    }
    let off = off.ok_or("every triangle is in one plane")?;
    let apart = corners(off.0, off.1)[off.2];
    let mut found = Vec::new();
    for (first, second) in [(0, 1), (0, 2), (1, 2)] {
        let third = 3 - first - second;
        for x in 0..3 {
            for y in 0..3 {
                let Some((on_first, on_second, on_third, position)) =
                    planar_meeting(shape[first], x, shape[second], y, shape[third], apart)?
                else {
                    continue;
                };
                let mut located = [Located::Face; 3];
                located[first] = on_first;
                located[second] = on_second;
                located[third] = on_third;
                found.push(Junction {
                    triangles,
                    located,
                    position,
                    how: How::Edges {
                        first: (triangles[first].0, triangles[first].1, x),
                        second: (triangles[second].0, triangles[second].1, y),
                        off,
                    },
                });
            }
        }
    }
    Ok(found)
}

/// Where edge `x` of one triangle crosses edge `y` of another in their plane, if that is inside or
/// on a third triangle in it, and where that is on each of the three.
fn planar_meeting(
    first: [&Coordinate<3>; 3],
    x: usize,
    second: [&Coordinate<3>; 3],
    y: usize,
    third: [&Coordinate<3>; 3],
    off: &Coordinate<3>,
) -> Result<Option<(Located, Located, Located, Coordinate<3>)>, &'static str> {
    let (p, q) = (first[x], first[(x + 1) % 3]);
    let (r, t) = (second[y], second[(y + 1) % 3]);
    let across_second = [p, q].map(|c| orient3d(r, t, off, c));
    let across_first = [r, t].map(|c| orient3d(p, q, off, c));
    if across_second[0] == across_second[1] || across_first[0] == across_first[1] {
        return Ok(None);
    }
    let point = Derived::Lpi {
        line: [p, q],
        plane: [r, t, off],
    };
    let Some(on_third) = locate(third, off, &point)? else {
        return Ok(None);
    };
    let on = |sides: [Ordering; 2], edge: usize| match sides {
        [Ordering::Equal, _] => Located::Vertex(edge),
        [_, Ordering::Equal] => Located::Vertex((edge + 1) % 3),
        _ => Located::Edge(edge),
    };
    Ok(Some((
        on(across_second, x),
        on(across_first, y),
        on_third,
        point.position()?,
    )))
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
        match locate(*triangle, off, &point)? {
            Some(found) => located[index] = found,
            None => return Ok(None),
        }
    }
    Ok(Some((located, point.position()?)))
}

/// Where a point in the plane of a triangle is on it, if it is on it, given a point `off` that is
/// not in that plane, so that the plane through each edge and `off` cuts the plane of the triangle
/// along that edge.
fn locate(
    triangle: [&Coordinate<3>; 3],
    off: &Coordinate<3>,
    point: &Derived,
) -> Result<Option<Located>, &'static str> {
    let mut on = Vec::new();
    for edge in 0..3 {
        let (u, v, t) = (
            triangle[edge],
            triangle[(edge + 1) % 3],
            triangle[(edge + 2) % 3],
        );
        let inside = orient3d(u, v, off, t);
        match side([u, v, off], point)? {
            Ordering::Equal => on.push(edge),
            found if found != inside => return Ok(None),
            _ => (),
        }
    }
    Ok(Some(match on[..] {
        [] => Located::Face,
        [edge] => Located::Edge(edge),
        [0, 1] => Located::Vertex(1),
        [1, 2] => Located::Vertex(2),
        [0, 2] => Located::Vertex(0),
        _ => return Err("a point is on all three edges of a triangle"),
    }))
}

/// The junctions at an edge of one of two triangles in one plane, where it meets a triangle that
/// crosses both, given for each pair of solids the pairs of their triangles that lie in one plane
/// and that cross each other.
///
/// The three planes do not meet in a point there, but the edge meets the plane of the third.
pub(super) fn edge_junctions<'a>(
    coplanar: &BTreeMap<[usize; 2], Vec<[usize; 2]>>,
    pairs: &BTreeMap<[usize; 2], FxHashSet<[usize; 2]>>,
    solids: usize,
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3],
) -> Result<Vec<Junction>, &'static str> {
    let mut crossing = FxHashMap::<(usize, usize, usize), Vec<usize>>::default();
    for (&[x, y], set) in pairs {
        for &[tx, ty] in set {
            crossing.entry((y, x, ty)).or_default().push(tx);
            crossing.entry((x, y, tx)).or_default().push(ty);
        }
    }
    let mut found = Vec::new();
    for (&[j, k], list) in coplanar {
        for &[b, c] in list {
            for i in (0..solids).filter(|&i| i != j && i != k) {
                let with_c: FxHashSet<usize> = crossing
                    .get(&(k, i, c))
                    .into_iter()
                    .flatten()
                    .copied()
                    .collect();
                let mut across: Vec<usize> = crossing
                    .get(&(j, i, b))
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|a| with_c.contains(a))
                    .collect();
                across.sort_unstable();
                for a in across {
                    let (plane, first, second) = (corners(i, a), corners(j, b), corners(k, c));
                    for (owner, other, solid, triangle, partner) in
                        [(first, second, j, b, (k, c)), (second, first, k, c, (j, b))]
                    {
                        for edge in 0..3 {
                            let Some(([on_a, on_owner, on_other], position)) =
                                edge_meeting(plane, owner, edge, other)?
                            else {
                                continue;
                            };
                            let mut places = [
                                (i, a, on_a),
                                (solid, triangle, on_owner),
                                (partner.0, partner.1, on_other),
                            ];
                            places.sort_unstable_by_key(|&(solid, ..)| solid);
                            found.push(Junction {
                                triangles: places.map(|(solid, triangle, _)| (solid, triangle)),
                                located: places.map(|(.., located)| located),
                                position,
                                how: How::Edge {
                                    solid,
                                    triangle,
                                    edge,
                                    plane: (i, a),
                                },
                            });
                        }
                    }
                }
            }
        }
    }
    Ok(found)
}

/// Where an edge of a triangle meets the plane of another, if that is inside or on the other
/// triangle, and the triangle in the plane of the first: where it is on each of the three.
fn edge_meeting(
    plane: [&Coordinate<3>; 3],
    owner: [&Coordinate<3>; 3],
    edge: usize,
    other: [&Coordinate<3>; 3],
) -> Result<Option<([Located; 3], Coordinate<3>)>, &'static str> {
    let (start, end) = (owner[edge], owner[(edge + 1) % 3]);
    let sides = [start, end].map(|corner| orient3d(plane[0], plane[1], plane[2], corner));
    if sides[0] == sides[1] {
        return Ok(None);
    }
    let point = Derived::Lpi {
        line: [start, end],
        plane,
    };
    let off = if sides[0] != Ordering::Equal {
        start
    } else {
        end
    };
    let Some(on_plane) = locate(plane, off, &point)? else {
        return Ok(None);
    };
    let along = plane
        .iter()
        .find(|corner| orient3d(owner[0], owner[1], owner[2], corner) != Ordering::Equal)
        .ok_or("two triangles that cross are in one plane")?;
    let Some(on_other) = locate(other, along, &point)? else {
        return Ok(None);
    };
    let on_owner = match sides {
        [Ordering::Equal, _] => Located::Vertex(edge),
        [_, Ordering::Equal] => Located::Vertex((edge + 1) % 3),
        _ => Located::Edge(edge),
    };
    Ok(Some(([on_plane, on_owner, on_other], point.position()?)))
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

/// A point at the end of a curve, and the features that name the end.
pub(super) type Link = (usize, [Feature; 2]);

type Arranged = (Vec<Point>, FxHashMap<Curve, Vec<usize>>, Vec<Link>);

/// The points, and the points strictly between the ends of every curve that has any, in order from
/// its start, where `curves` gives the ends of every curve and `feature` the feature of a solid
/// that a place on one of its triangles is.
///
/// A point is at the end of a curve when it is on the same features as the end is, and is left out
/// of those between, but is linked to the end, which is named by only those features. Junctions that are the same point, because a
/// fourth surface meets the same three there, are one point. Fails where points on a curve cannot
/// be put in order, which is where one is too near an end of the curve for its rounded position to
/// say which side of the other surface it is on.
pub(super) fn arrange<'a>(
    junctions: &[Junction],
    curves: &FxHashMap<Curve, Ends>,
    feature: impl Fn(usize, usize, Located) -> Feature,
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3] + Copy,
) -> Result<Arranged, &'static str> {
    let mut through = FxHashMap::<Curve, Vec<usize>>::default();
    for (index, junction) in junctions.iter().enumerate() {
        for (n, &(first, a)) in junction.triangles.iter().enumerate() {
            for &(second, b) in &junction.triangles[n + 1..] {
                let curve = ([first, second], a, b);
                if curves.contains_key(&curve) {
                    through.entry(curve).or_default().push(index);
                }
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
    // The same lowest feature of every solid is the same point, however it was found.
    let mut named = FxHashMap::<Vec<Feature>, usize>::default();
    for (index, junction) in junctions.iter().enumerate() {
        let features: Vec<Feature> = junction
            .triangles
            .iter()
            .zip(&junction.located)
            .map(|(&(solid, triangle), &located)| feature(solid, triangle, located))
            .collect();
        match named.get(&features) {
            Some(&first) => same.join(&first, &index),
            None => {
                named.insert(features, index);
            }
        }
    }
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
    let mut numbered = FxHashMap::<usize, usize>::default();
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
    let mut links = Vec::new();
    let mut arranged = FxHashMap::default();
    for (curve, found) in through {
        for &x in &found {
            if let Some(end) = end_of(&curve, x) {
                links.push((
                    numbered[&same.find(&x)],
                    curves[&curve].features[end].clone(),
                ));
            }
        }
        let mut roots: Vec<usize> = found
            .iter()
            .filter(|&&x| end_of(&curve, x).is_none())
            .map(|x| same.find(x))
            .collect();
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
        arranged.insert(curve, roots.iter().map(|root| numbered[root]).collect());
    }
    Ok((points, arranged, links))
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
    corners: impl Fn(usize, usize) -> [&'a Coordinate<3>; 3] + Copy,
) -> Result<Ordering, &'static str> {
    let point = junctions[y].derived(corners);
    let mut cut = None;
    for plane in junctions[x].across(&curve.0, corners)? {
        let found = side(plane, &point)?;
        if found != Ordering::Equal {
            cut = Some((plane, found));
            break;
        }
    }
    let Some((plane, found)) = cut else {
        return Ok(Ordering::Equal);
    };
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
