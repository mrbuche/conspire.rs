//! Splitting the triangles of two tessellations along where they meet.

#[cfg(test)]
mod test;

use super::{
    Contacts, Feature, Key, Side,
    frame::{frame, own as own_feature},
    triangulate::triangulate,
};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Connectivity, Mesh, Tessellation},
        predicates::Cutter,
    },
    math::Tensor,
};
use std::collections::{BTreeMap, HashMap};

/// What meets one triangle: points, each with a rounded position, and the segments between them.
#[derive(Default)]
struct Piece {
    points: Vec<(Key, Coordinate<3>)>,
    segments: Vec<[Key; 2]>,
}

impl Piece {
    fn add(&mut self, keys: &[Key; 2], segment: &[Coordinate<3>; 2]) {
        keys.iter()
            .zip(segment)
            .for_each(|(key, end)| self.points.push((key.clone(), end.clone())));
        if keys[0] != keys[1] {
            self.segments.push([keys[0].clone(), keys[1].clone()]);
        }
    }
}

impl Tessellation {
    /// This tessellation and the other, each with its triangles split along the curves where the
    /// two meet, and along the edges of the other where they overlap in one plane.
    ///
    /// The vertices where the two meet are new vertices of each, shared by the triangles around
    /// them, so a closed surface stays closed. Fails where exactness cannot be kept: where a
    /// point that is rounded falls outside the triangle it belongs to, or where segments cross
    /// away from a point.
    pub fn corefine(&self, other: &Tessellation) -> Result<[Tessellation; 2], &'static str> {
        let contacts = self.contacts(other)?;
        Ok([
            split(self, other, Side::First, &contacts)?,
            split(other, self, Side::Second, &contacts)?,
        ])
    }
}

fn split(
    own: &Tessellation,
    other: &Tessellation,
    side: Side,
    contacts: &Contacts,
) -> Result<Tessellation, &'static str> {
    let mut pieces: BTreeMap<usize, Piece> = BTreeMap::new();
    for crossing in &contacts.crossings {
        let triangle = match side {
            Side::First => crossing.first,
            Side::Second => crossing.second,
        };
        pieces
            .entry(triangle)
            .or_default()
            .add(&crossing.keys, &crossing.segment);
    }
    for constraint in contacts.constraints.iter().filter(|c| c.side == side) {
        pieces
            .entry(constraint.triangle)
            .or_default()
            .add(&constraint.keys, &constraint.segment);
    }
    let mut around = HashMap::<[usize; 2], Vec<usize>>::new();
    for (index, nodes) in own.mesh().connectivities().iter().flatten().enumerate() {
        (0..3).for_each(|edge| {
            let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        })
    }
    let shared: Vec<(usize, Key, Coordinate<3>)> = pieces
        .iter()
        .flat_map(|(&triangle, piece)| {
            let around = &around;
            piece.points.iter().flat_map(move |(key, position)| {
                let neighbours = match own_feature(key, side) {
                    Feature::Edge(edge) => around[edge].as_slice(),
                    _ => &[],
                };
                neighbours
                    .iter()
                    .filter(move |&&neighbour| neighbour != triangle)
                    .map(move |&neighbour| (neighbour, key.clone(), position.clone()))
            })
        })
        .collect();
    for (triangle, key, position) in shared {
        pieces
            .entry(triangle)
            .or_default()
            .points
            .push((key, position));
    }
    let coordinates = own.mesh().coordinates();
    let mut vertices: Vec<Coordinate<3>> = coordinates.iter().cloned().collect();
    let other_coordinates = other.mesh().coordinates();
    let other_triangles: Vec<&[usize]> = other.mesh().connectivities().iter().flatten().collect();
    let cutter = |feature: &Feature| match feature {
        Feature::Face(i) => {
            let nodes = other_triangles[*i];
            Cutter::Face([
                &other_coordinates[nodes[0]],
                &other_coordinates[nodes[1]],
                &other_coordinates[nodes[2]],
            ])
        }
        Feature::Edge([a, b]) => Cutter::Edge([&other_coordinates[*a], &other_coordinates[*b]]),
        Feature::Vertex(node) => Cutter::Vertex(&other_coordinates[*node]),
    };
    let mut named = HashMap::<Key, usize>::new();
    let mut faces: Vec<[usize; 3]> = Vec::new();
    for (index, nodes) in own.mesh().connectivities().iter().flatten().enumerate() {
        let nodes = [nodes[0], nodes[1], nodes[2]];
        let Some(piece) = pieces.get(&index) else {
            faces.push(nodes);
            continue;
        };
        let framed = frame(
            nodes.map(|node| &coordinates[node]),
            nodes,
            side,
            &piece.points,
            &piece.segments,
            cutter,
        )?;
        let positions: HashMap<&Key, &Coordinate<3>> = piece
            .points
            .iter()
            .map(|(key, position)| (key, position))
            .collect();
        let ids: Vec<usize> = framed
            .keys
            .iter()
            .map(|key| {
                *named.entry(key.clone()).or_insert_with(|| {
                    vertices.push(positions[key].clone());
                    vertices.len() - 1
                })
            })
            .collect();
        let node = |point: usize| {
            if point < 3 {
                nodes[point]
            } else {
                ids[point - 3]
            }
        };
        faces.extend(
            triangulate(&framed.input)?
                .into_iter()
                .map(|[a, b, c]| [node(a), node(b), node(c)]),
        );
    }
    Ok(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        vertices.into(),
    ))
    .into())
}
