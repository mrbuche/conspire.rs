#[cfg(test)]
mod test;

use super::{
    Brep, D,
    inside::{directions, encloses},
};
use crate::{
    geometry::{
        CoordinatesRef, Direction,
        bbox::BoundingBox,
        mesh::{Class, Mesh},
    },
    math::Scalar,
};
use std::{
    array::from_fn,
    collections::{HashMap, hash_map::Entry},
};

const EPSILON: Scalar = 1.0e-12;

impl Brep {
    /// Classifies every cell of `mesh` against this solid: `Cut` if the surface
    /// passes through it, else `Inside` or `Outside`. Planar faces only.
    pub fn classify(&self, mesh: &Mesh<D>) -> Result<Vec<Class>, &'static str> {
        let faces = self
            .faces
            .iter()
            .map(|face| self.planar_face(face))
            .collect::<Result<Vec<_>, _>>()?;
        let directions = directions();
        let coordinates = mesh.coordinates();
        let number_of_elements = mesh.number_of_elements();

        let mut cut = vec![false; number_of_elements];
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); number_of_elements];
        let mut owner: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut index = 0;
        for block in mesh.iter() {
            for element in block.iter() {
                let nodes = block.element_nodes(element);
                let bbox = BoundingBox::from(
                    nodes
                        .iter()
                        .map(|&node| &coordinates[node])
                        .collect::<CoordinatesRef<'_, D>>(),
                );
                cut[index] = faces.iter().any(|face| {
                    bbox.overlaps(&face.aabb)
                        && box_hits_polygon(&bbox, &face.normal, &face.outline)
                });
                if !cut[index] {
                    for mut face in block.element_faces(element) {
                        face.sort_unstable();
                        match owner.entry(face) {
                            Entry::Occupied(entry) => {
                                let other = *entry.get();
                                neighbors[index].push(other);
                                neighbors[other].push(index);
                            }
                            Entry::Vacant(slot) => {
                                slot.insert(index);
                            }
                        }
                    }
                }
                index += 1;
            }
        }

        let centroids = mesh.centroids();
        let mut classes: Vec<Class> = cut
            .iter()
            .map(|&flag| if flag { Class::Cut } else { Class::Outside })
            .collect();
        let mut visited = cut;
        let mut stack = Vec::new();
        for seed in 0..number_of_elements {
            if !visited[seed] {
                let class = if encloses(&centroids[seed], &faces, &directions) {
                    Class::Inside
                } else {
                    Class::Outside
                };
                visited[seed] = true;
                stack.push(seed);
                while let Some(current) = stack.pop() {
                    classes[current] = class;
                    for &next in &neighbors[current] {
                        if !visited[next] {
                            visited[next] = true;
                            stack.push(next);
                        }
                    }
                }
            }
        }
        Ok(classes)
    }
}

/// Separating-axis test between an axis-aligned box and a planar polygon: box
/// axes, the polygon normal, and each box-axis crossed with a polygon edge. May
/// over-report for a non-convex polygon, which only over-marks `Cut`.
fn box_hits_polygon(bbox: &BoundingBox<D>, normal: &Direction<D>, polygon: &[[Scalar; D]]) -> bool {
    let center: [Scalar; D] =
        from_fn(|k| 0.5 * (bbox.minimum()[k].value() + bbox.maximum()[k].value()));
    let half: [Scalar; D] =
        from_fn(|k| 0.5 * (bbox.maximum()[k].value() - bbox.minimum()[k].value()));
    let vertices: Vec<[Scalar; D]> = polygon
        .iter()
        .map(|point| from_fn(|k| point[k] - center[k]))
        .collect();

    for k in 0..D {
        let (low, high) = extent(vertices.iter().map(|vertex| vertex[k]));
        if low > half[k] || high < -half[k] {
            return false;
        }
    }

    let normal: [Scalar; D] = from_fn(|k| normal[k].value());
    let radius: Scalar = (0..D).map(|k| half[k] * normal[k].abs()).sum();
    if dot(normal, vertices[0]).abs() > radius {
        return false;
    }

    let count = vertices.len();
    for i in 0..count {
        let edge: [Scalar; D] = from_fn(|k| vertices[(i + 1) % count][k] - vertices[i][k]);
        for k in 0..D {
            let mut axis = [0.0; D];
            axis[k] = 1.0;
            let axis = cross(axis, edge);
            if dot(axis, axis) < EPSILON {
                continue;
            }
            let radius: Scalar = (0..D).map(|j| half[j] * axis[j].abs()).sum();
            let (low, high) = extent(vertices.iter().map(|vertex| dot(axis, *vertex)));
            if low > radius || high < -radius {
                return false;
            }
        }
    }
    true
}

fn extent(values: impl Iterator<Item = Scalar>) -> (Scalar, Scalar) {
    values.fold(
        (Scalar::INFINITY, Scalar::NEG_INFINITY),
        |(low, high), value| (low.min(value), high.max(value)),
    )
}

fn dot(a: [Scalar; D], b: [Scalar; D]) -> Scalar {
    (0..D).map(|k| a[k] * b[k]).sum()
}

fn cross(a: [Scalar; D], b: [Scalar; D]) -> [Scalar; D] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
