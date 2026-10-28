#[cfg(test)]
mod test;

use super::Tessellation;
use crate::{
    geometry::{Coordinate, CoordinatesRef, Direction},
    math::{CrossProduct, Quantity, Tensor},
    units::Length,
};

type Triangle<'a> = [&'a Coordinate<3>; 3];

/// A segment along which a triangle of one tessellation crosses a triangle of another.
pub struct Crossing {
    pub first: usize,
    pub second: usize,
    pub segment: [Coordinate<3>; 2],
}

impl Tessellation {
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
        for (first, nodes) in triangles.into_iter().enumerate() {
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
                if let Some(segment) = crossing(triangle, other_triangle)? {
                    crossings.push(Crossing {
                        first,
                        second,
                        segment,
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
) -> Result<Option<[Coordinate<3>; 2]>, &'static str> {
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
    let (Some(a), Some(b)) = (
        interval(first, &to_first, &line),
        interval(second, &to_second, &line),
    ) else {
        return Ok(None);
    };
    let start = if a[0].0 >= b[0].0 { &a[0] } else { &b[0] };
    let end = if a[1].0 <= b[1].0 { &a[1] } else { &b[1] };
    Ok((start.0 <= end.0).then(|| [start.1.clone(), end.1.clone()]))
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

/// The two ends, with positions along the line, where a triangle crosses the other plane.
fn interval(
    triangle: Triangle,
    offsets: &[Quantity<Length>; 3],
    line: &Direction<3>,
) -> Option<[(Quantity<Length>, Coordinate<3>); 2]> {
    let zero = Quantity::default();
    let mut ends = Vec::with_capacity(3);
    (0..3).for_each(|index| {
        let next = (index + 1) % 3;
        if offsets[index] == zero {
            ends.push(triangle[index].clone())
        } else if (offsets[index] < zero) && (offsets[next] > zero)
            || (offsets[index] > zero) && (offsets[next] < zero)
        {
            let fraction = offsets[index] / (offsets[index] - offsets[next]);
            ends.push(triangle[index] + (triangle[next] - triangle[index]) * fraction)
        }
    });
    let [a, b] = ends.try_into().ok()?;
    let (a, b) = ((&a * line, a), (&b * line, b));
    Some(if a.0 <= b.0 { [a, b] } else { [b, a] })
}
