#[cfg(test)]
mod test;

use super::{Triangle, pierce::Local};
use crate::geometry::{
    Coordinate,
    predicates::{orient2d, orient2d_estimate},
};
use core::cmp::Ordering;

/// An end of the part of a segment that lies inside a triangle.
pub(super) struct Cut {
    /// The position of the end, rounded: only the features it lies on are exact.
    pub point: Coordinate<3>,
    /// What the end lies on in the triangle.
    pub triangle: Local,
    /// What the end lies on in the segment: its start `Vertex(0)`, its finish `Vertex(1)`, or its
    /// inside `Edge(0)`.
    pub segment: Local,
}

/// The part of the segment from `start` to `finish` inside the triangle, as its two ends in order
/// along the segment, if it has any length.
///
/// The segment and triangle are in one plane, and `axis` is a coordinate along which the normal of
/// the plane is not zero. Which side of an edge a point is on is decided exactly, so each end is
/// named by the vertex or edge of the triangle it lies on even where it falls exactly on one.
pub(super) fn clip(
    start: &Coordinate<3>,
    finish: &Coordinate<3>,
    triangle: Triangle,
    axis: usize,
) -> Option<[Cut; 2]> {
    let turn = orient2d(triangle[0], triangle[1], triangle[2], axis);
    if turn == Ordering::Equal {
        return None;
    }
    let sides = |point: &Coordinate<3>| -> [Ordering; 3] {
        [0, 1, 2].map(|edge| {
            let side = orient2d(triangle[edge], triangle[(edge + 1) % 3], point, axis);
            if turn == Ordering::Less {
                side.reverse()
            } else {
                side
            }
        })
    };
    let (sides_start, sides_finish) = (sides(start), sides(finish));
    let first = end(
        (start, &sides_start, Local::Vertex(0)),
        (finish, &sides_finish),
        triangle,
        axis,
    )?;
    let last = end(
        (finish, &sides_finish, Local::Vertex(1)),
        (start, &sides_start),
        triangle,
        axis,
    )?;
    (first.triangle != last.triangle || first.segment != last.segment).then_some([first, last])
}

/// The end of the clipped segment on the side of `near`, looking toward `far`.
fn end(
    near: (&Coordinate<3>, &[Ordering; 3], Local),
    far: (&Coordinate<3>, &[Ordering; 3]),
    triangle: Triangle,
    axis: usize,
) -> Option<Cut> {
    let (point, sides, name) = near;
    if let Some(on) = locate(sides) {
        return Some(Cut {
            point: point.clone(),
            triangle: on,
            segment: name,
        });
    }
    let (other, sides_other) = far;
    for edge in 0..3 {
        if sides[edge] != Ordering::Less || sides_other[edge] == Ordering::Less {
            continue;
        }
        let next = (edge + 1) % 3;
        let [along_edge, along_next] =
            [edge, next].map(|vertex| orient2d(point, other, triangle[vertex], axis));
        if along_edge == along_next && along_edge != Ordering::Equal {
            continue;
        }
        let on = if along_edge == Ordering::Equal {
            Local::Vertex(edge)
        } else if along_next == Ordering::Equal {
            Local::Vertex(next)
        } else {
            Local::Edge(edge)
        };
        let (crossing, segment) = if sides_other[edge] == Ordering::Equal {
            (other.clone(), far_name(name))
        } else {
            let position = match on {
                Local::Vertex(vertex) => triangle[vertex].clone(),
                _ => {
                    let [above_near, above_far] = [point, other]
                        .map(|end| orient2d_estimate(triangle[edge], triangle[next], end, axis));
                    let fraction = (above_near / (above_near - above_far)).clamp(0.0, 1.0);
                    point + (other - point) * fraction
                }
            };
            (position, Local::Edge(0))
        };
        return Some(Cut {
            point: crossing,
            triangle: on,
            segment,
        });
    }
    None
}

/// The other end of the segment from the one named.
fn far_name(name: Local) -> Local {
    match name {
        Local::Vertex(0) => Local::Vertex(1),
        _ => Local::Vertex(0),
    }
}

/// What a point with these sides of the edges lies on in the triangle, if it is not outside.
fn locate(sides: &[Ordering; 3]) -> Option<Local> {
    if sides.contains(&Ordering::Less) {
        return None;
    }
    let zeros: Vec<usize> = (0..3)
        .filter(|&edge| sides[edge] == Ordering::Equal)
        .collect();
    match zeros[..] {
        [] => Some(Local::Face),
        [edge] => Some(Local::Edge(edge)),
        [_, _] => (0..3)
            .find(|&edge| {
                sides[edge] == Ordering::Equal && sides[(edge + 1) % 3] == Ordering::Equal
            })
            .map(|edge| Local::Vertex((edge + 1) % 3)),
        _ => None,
    }
}
