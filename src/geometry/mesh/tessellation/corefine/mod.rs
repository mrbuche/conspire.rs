#[cfg(test)]
mod test;

mod chain;

pub use chain::Polyline;

use super::Tessellation;
use crate::{
    geometry::{Coordinate, CoordinatesRef, Direction},
    math::{CrossProduct, Quantity, Tensor},
    units::Length,
};
use chain::chain;

type Triangle<'a> = [&'a Coordinate<3>; 3];

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub(super) enum Side {
    First,
    Second,
}

#[derive(Clone, Copy)]
enum Origin {
    /// A vertex of the triangle.
    Vertex(usize),
    /// The edge from a vertex to the next.
    Edge(usize),
}

pub(super) struct End {
    point: Coordinate<3>,
    side: Side,
    origin: Origin,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub(super) enum Key {
    Vertex {
        side: Side,
        node: usize,
        other: usize,
    },
    Edge {
        side: Side,
        nodes: [usize; 2],
        other: usize,
    },
}

/// A segment along which a triangle of one tessellation crosses a triangle of another.
pub struct Crossing {
    pub first: usize,
    pub second: usize,
    pub segment: [Coordinate<3>; 2],
    pub(super) keys: [Key; 2],
}

impl Tessellation {
    /// The curves along which this tessellation intersects another.
    pub fn intersection_curves(&self, other: &Tessellation) -> Result<Vec<Polyline>, &'static str> {
        chain(&self.crossings(other)?)
    }
    /// Every segment along which this tessellation crosses another.
    ///
    /// Fails on coplanar or degenerate triangle pairs, which have no crossing segment.
    pub fn crossings(&self, other: &Tessellation) -> Result<Vec<Crossing>, &'static str> {
        let (coordinates, other_coordinates) =
            (self.mesh().coordinates(), other.mesh().coordinates());
        let triangles: Vec<&[usize]> = self.mesh().connectivities().iter().flatten().collect();
        let other_triangles: Vec<&[usize]> =
            other.mesh().connectivities().iter().flatten().collect();
        let other_bvh = other.bvh();
        let mut crossings = Vec::new();
        for (first, nodes) in triangles.iter().enumerate() {
            let triangle = [
                &coordinates[nodes[0]],
                &coordinates[nodes[1]],
                &coordinates[nodes[2]],
            ];
            let bbox = triangle
                .into_iter()
                .collect::<CoordinatesRef<'_, 3>>()
                .into();
            for second in other_bvh.overlapping(&bbox) {
                let nodes = other_triangles[second];
                let other_triangle = [
                    &other_coordinates[nodes[0]],
                    &other_coordinates[nodes[1]],
                    &other_coordinates[nodes[2]],
                ];
                if let Some(ends) = crossing(triangle, other_triangle)? {
                    let key = |end: &End| {
                        let (nodes, other) = match end.side {
                            Side::First => (triangles[first], second),
                            Side::Second => (other_triangles[second], first),
                        };
                        match end.origin {
                            Origin::Vertex(index) => Key::Vertex {
                                side: end.side,
                                node: nodes[index],
                                other,
                            },
                            Origin::Edge(index) => {
                                let (a, b) = (nodes[index], nodes[(index + 1) % 3]);
                                Key::Edge {
                                    side: end.side,
                                    nodes: [a.min(b), a.max(b)],
                                    other,
                                }
                            }
                        }
                    };
                    crossings.push(Crossing {
                        first,
                        second,
                        keys: [key(&ends[0]), key(&ends[1])],
                        segment: ends.map(|end| end.point),
                    });
                }
            }
        }
        Ok(crossings)
    }
}

pub(super) fn crossing(
    first: Triangle,
    second: Triangle,
) -> Result<Option<[End; 2]>, &'static str> {
    let normals = [first, second].map(normal);
    let (Some(first_normal), Some(second_normal)) = (&normals[0], &normals[1]) else {
        return Err("degenerate triangle");
    };
    let to_second = offsets(first[0], first_normal, second);
    let to_first = offsets(second[0], second_normal, first);
    if to_second
        .iter()
        .all(|&offset| offset == Quantity::default())
    {
        return Err("coplanar triangles");
    }
    if separated(&to_second) || separated(&to_first) {
        return Ok(None);
    }
    let line: Direction<3> = first_normal.cross(second_normal).normalized();
    let (Some([a_start, a_end]), Some([b_start, b_end])) = (
        interval(first, &to_first, &line, Side::First),
        interval(second, &to_second, &line, Side::Second),
    ) else {
        return Ok(None);
    };
    let start = if a_start.0 >= b_start.0 {
        a_start
    } else {
        b_start
    };
    let end = if a_end.0 <= b_end.0 { a_end } else { b_end };
    Ok((start.0 <= end.0).then_some([start.1, end.1]))
}

fn normal(triangle: Triangle) -> Option<Direction<3>> {
    let area = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    (area.norm() > Quantity::default()).then(|| area.normalized())
}

fn offsets(
    origin: &Coordinate<3>,
    normal: &Direction<3>,
    triangle: Triangle,
) -> [Quantity<Length>; 3] {
    triangle.map(|vertex| (vertex - origin) * normal)
}

fn separated(offsets: &[Quantity<Length>; 3]) -> bool {
    let zero = Quantity::default();
    offsets.iter().all(|&offset| offset > zero) || offsets.iter().all(|&offset| offset < zero)
}

fn interval(
    triangle: Triangle,
    offsets: &[Quantity<Length>; 3],
    line: &Direction<3>,
    side: Side,
) -> Option<[(Quantity<Length>, End); 2]> {
    let zero = Quantity::default();
    let mut ends = Vec::with_capacity(3);
    (0..3).for_each(|index| {
        let next = (index + 1) % 3;
        if offsets[index] == zero {
            ends.push((triangle[index].clone(), Origin::Vertex(index)))
        } else if (offsets[index] < zero) && (offsets[next] > zero)
            || (offsets[index] > zero) && (offsets[next] < zero)
        {
            let fraction = offsets[index] / (offsets[index] - offsets[next]);
            ends.push((
                triangle[index] + (triangle[next] - triangle[index]) * fraction,
                Origin::Edge(index),
            ))
        }
    });
    let [a, b] = ends.try_into().ok()?;
    let [a, b] = [a, b].map(|(point, origin)| {
        (
            &point * line,
            End {
                point,
                side,
                origin,
            },
        )
    });
    Some(if a.0 <= b.0 { [a, b] } else { [b, a] })
}
