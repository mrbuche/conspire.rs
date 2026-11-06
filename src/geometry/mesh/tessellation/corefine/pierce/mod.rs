#[cfg(test)]
mod test;

use super::Triangle;
use crate::geometry::{
    Coordinate,
    predicates::{collinear, orient3d, orient3d_estimate},
};
use core::cmp::Ordering;

/// The lowest-dimensional feature of a triangle that contains a point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Local {
    /// A vertex of the triangle.
    Vertex(usize),
    /// The edge from a vertex to the next.
    Edge(usize),
    /// The inside of the triangle.
    Face,
}

/// An end of a crossing segment, and the feature of each triangle it lies on.
pub(super) struct End {
    pub point: Coordinate<3>,
    pub first: Local,
    pub second: Local,
}

/// How two triangles meet.
pub(super) enum Meeting {
    /// They do not meet in a segment: they are separated, or touch at a single point.
    Apart,
    /// They lie in one plane, so they meet in an area, or not at all, rather than in a segment.
    Coplanar,
    /// They cross along a segment.
    Segment([End; 2]),
}

/// How two triangles meet, and the segment along which they cross if they do.
///
/// Degenerate triangles, whose vertices are collinear, are an error. Which side of a plane a
/// vertex is on, and where a boundary point lies in the other triangle, are decided exactly, so a
/// segment end is named by the features it lies on even where it falls exactly on an edge or a
/// vertex. Only the coordinates of an end are rounded.
pub(super) fn crossing(first: Triangle, second: Triangle) -> Result<Meeting, &'static str> {
    if collinear(first[0], first[1], first[2]) || collinear(second[0], second[1], second[2]) {
        return Err("degenerate triangle");
    }
    let sides_first = sides(second, first);
    let sides_second = sides(first, second);
    if sides_first.iter().all(|&side| side == Ordering::Equal) {
        return Ok(Meeting::Coplanar);
    }
    if separated(&sides_first) || separated(&sides_second) {
        return Ok(Meeting::Apart);
    }
    let (Some(ends_first), Some(ends_second)) = (
        plane_crossings(&sides_first),
        plane_crossings(&sides_second),
    ) else {
        return Ok(Meeting::Apart);
    };
    let mut ends = Vec::<End>::with_capacity(4);
    for local in ends_first {
        let (start, finish, point) = boundary_point(first, second, &sides_first, local);
        if let Some(on) = locate(start, finish, second) {
            ends.push(End {
                point,
                first: local,
                second: on,
            });
        }
    }
    for local in ends_second {
        let (start, finish, point) = boundary_point(second, first, &sides_second, local);
        if let Some(on) = locate(start, finish, first) {
            ends.push(End {
                point,
                first: on,
                second: local,
            });
        }
    }
    let mut distinct = Vec::<End>::with_capacity(2);
    for end in ends {
        if !distinct
            .iter()
            .any(|other| other.first == end.first && other.second == end.second)
        {
            distinct.push(end);
        }
    }
    match <[End; 2]>::try_from(distinct) {
        Ok(ends) => Ok(Meeting::Segment(ends)),
        Err(distinct) if distinct.len() > 2 => Err("a crossing has more than two ends"),
        Err(_) => Ok(Meeting::Apart),
    }
}

/// Which side of the plane of `plane` each vertex of `points` is on.
fn sides(plane: Triangle, points: Triangle) -> [Ordering; 3] {
    points.map(|point| orient3d(plane[0], plane[1], plane[2], point))
}

fn separated(sides: &[Ordering; 3]) -> bool {
    sides.iter().all(|&side| side == Ordering::Greater)
        || sides.iter().all(|&side| side == Ordering::Less)
}

/// The vertices on the plane and the edges crossing it, which are where the boundary of the
/// triangle meets the plane; `None` unless that is two features, so a segment.
fn plane_crossings(sides: &[Ordering; 3]) -> Option<[Local; 2]> {
    let mut crossings = Vec::with_capacity(3);
    for index in 0..3 {
        let next = (index + 1) % 3;
        if sides[index] == Ordering::Equal {
            crossings.push(Local::Vertex(index));
        } else if sides[next] == sides[index].reverse() {
            crossings.push(Local::Edge(index));
        }
    }
    crossings.try_into().ok()
}

/// A segment of the boundary of `triangle` that meets the plane of `plane` at the feature, and the
/// point where it does.
fn boundary_point<'a>(
    triangle: Triangle<'a>,
    plane: Triangle,
    sides: &[Ordering; 3],
    local: Local,
) -> (&'a Coordinate<3>, &'a Coordinate<3>, Coordinate<3>) {
    match local {
        Local::Edge(index) => {
            let (start, finish) = (triangle[index], triangle[(index + 1) % 3]);
            let [above_start, above_finish] =
                [start, finish].map(|point| orient3d_estimate(plane[0], plane[1], plane[2], point));
            let fraction = (above_start / (above_start - above_finish)).clamp(0.0, 1.0);
            (start, finish, start + (finish - start) * fraction)
        }
        Local::Vertex(index) => {
            let off = (0..3)
                .find(|&other| sides[other] != Ordering::Equal)
                .unwrap_or(index);
            (triangle[index], triangle[off], triangle[index].clone())
        }
        Local::Face => unreachable!("the boundary of a triangle is not its inside"),
    }
}

/// The feature of `triangle` that the segment from `start` to `finish` passes through where it
/// meets the plane of the triangle, if it passes through the triangle at all.
///
/// The three signs say which side of each edge line the meeting point is on, as seen along the
/// segment. A point inside has none of one sign and some of the other, and a zero puts it on that
/// edge line.
fn locate(start: &Coordinate<3>, finish: &Coordinate<3>, triangle: Triangle) -> Option<Local> {
    let signs: [Ordering; 3] =
        [0, 1, 2].map(|edge| orient3d(start, finish, triangle[edge], triangle[(edge + 1) % 3]));
    if signs.contains(&Ordering::Less) && signs.contains(&Ordering::Greater) {
        return None;
    }
    let zeros: Vec<usize> = (0..3)
        .filter(|&edge| signs[edge] == Ordering::Equal)
        .collect();
    match zeros[..] {
        [] => Some(Local::Face),
        [edge] => Some(Local::Edge(edge)),
        [_, _] => (0..3)
            .find(|&edge| {
                signs[edge] == Ordering::Equal && signs[(edge + 1) % 3] == Ordering::Equal
            })
            .map(|edge| Local::Vertex((edge + 1) % 3)),
        _ => None,
    }
}
