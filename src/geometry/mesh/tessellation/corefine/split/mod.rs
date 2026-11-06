//! Splitting the triangles of tessellations along where they meet.

#[cfg(test)]
mod test;

use super::{Contacts, Feature, Key, Side, frame::frame, triangulate::triangulate_constrained};
use crate::math::{FxHashMap, FxHashSet};
use crate::{
    geometry::{Coordinate, mesh::Tessellation, predicates::Cutter},
    math::Tensor,
};
use alloc::collections::BTreeMap;

/// What meets one triangle: points, each with a rounded position, and the segments between them.
pub(super) struct Piece<K> {
    pub points: Vec<(K, Coordinate<3>)>,
    pub segments: Vec<[K; 2]>,
}

impl<K> Default for Piece<K> {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            segments: Vec::new(),
        }
    }
}

impl<K: Clone + PartialEq> Piece<K> {
    pub fn add(&mut self, keys: &[K; 2], segment: &[Coordinate<3>; 2]) {
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
            split(self, other, Side::First, &contacts)?.tessellation,
            split(other, self, Side::Second, &contacts)?.tessellation,
        ])
    }
}

/// A tessellation split along where another meets it.
pub(super) struct Split {
    pub tessellation: Tessellation,
    /// The triangle of the tessellation that was split, for each triangle of the result.
    pub parents: Vec<usize>,
    /// The edges, as pairs of vertices with the smaller first, along which the other meets it.
    pub curves: FxHashSet<[usize; 2]>,
    /// The vertex of the result at each point that was added, by the key that names it.
    pub named: FxHashMap<Key, usize>,
}

pub(super) fn split(
    own: &Tessellation,
    other: &Tessellation,
    side: Side,
    contacts: &Contacts,
) -> Result<Split, &'static str> {
    let mut pieces = BTreeMap::<usize, Piece<Key>>::new();
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
    let faces: Vec<[usize; 3]> = own
        .mesh()
        .connectivities()
        .iter()
        .flatten()
        .map(|nodes| [nodes[0], nodes[1], nodes[2]])
        .collect();
    let mut named = FxHashMap::<Key, usize>::default();
    let refined = refine(
        &faces,
        |node| &coordinates[node],
        pieces,
        |key| key.on(side),
        |key| vec![cutter(key.across(side))],
        |key, position| {
            *named.entry(key.clone()).or_insert_with(|| {
                vertices.push(position.clone());
                vertices.len() - 1
            })
        },
    )?;
    Ok(Split {
        tessellation: (refined.faces, vertices).into(),
        parents: refined.parents,
        curves: refined.curves,
        named,
    })
}

/// Triangles split along what meets them.
pub(super) struct Refined {
    pub faces: Vec<[usize; 3]>,
    /// The triangle that each was split from.
    pub parents: Vec<usize>,
    /// The edges, as pairs of vertices with the smaller first, that the segments became.
    pub curves: FxHashSet<[usize; 2]>,
}

/// Split triangles of one surface into the pieces that what meets them cuts them into.
///
/// The triangles name their corners by `faces`, which are also what the features of the keys of
/// `pieces` name, and `position` gives where a corner is. Each key has a `feature`, the lowest
/// feature of the surface that its point is on, and `cutters`, which cut an edge at the point. A
/// point that is not a corner is named by `name`, from its key and where it is rounded to, with
/// the vertex it makes. A point on an edge is given to every triangle around the edge, including
/// those that nothing else meets, so that triangles agree on how an edge is divided.
pub(super) fn refine<'c, 'a, K: Ord + Clone>(
    faces: &[[usize; 3]],
    position: impl Fn(usize) -> &'c Coordinate<3>,
    mut pieces: BTreeMap<usize, Piece<K>>,
    feature: impl Fn(&K) -> &Feature,
    cutters: impl Fn(&K) -> Vec<Cutter<'a>>,
    mut name: impl FnMut(&K, &Coordinate<3>) -> usize,
) -> Result<Refined, &'static str> {
    share(faces, &mut pieces, &feature);
    let mut refined = Refined {
        faces: Vec::new(),
        parents: Vec::new(),
        curves: FxHashSet::default(),
    };
    for (index, &nodes) in faces.iter().enumerate() {
        let Some(piece) = pieces.get(&index) else {
            refined.faces.push(nodes);
            refined.parents.push(index);
            continue;
        };
        let framed = frame(
            nodes.map(&position),
            nodes,
            &piece.points,
            &piece.segments,
            &feature,
            &cutters,
        )?;
        let positions: BTreeMap<&K, &Coordinate<3>> = piece
            .points
            .iter()
            .map(|(key, point)| (key, point))
            .collect();
        let ids: Vec<usize> = framed
            .keys
            .iter()
            .map(|key| name(key, positions[key]))
            .collect();
        let node = |point: usize| {
            if point < 3 {
                nodes[point]
            } else {
                ids[point - 3]
            }
        };
        let (triangles, constrained) = triangulate_constrained(&framed.input)?;
        refined.parents.extend(triangles.iter().map(|_| index));
        refined.faces.extend(
            triangles
                .into_iter()
                .map(|[a, b, c]| [node(a), node(b), node(c)]),
        );
        refined.curves.extend(constrained.into_iter().map(|[a, b]| {
            let (a, b) = (node(a), node(b));
            [a.min(b), a.max(b)]
        }));
    }
    Ok(refined)
}

/// Gives every triangle around an edge the points on it, so that triangles agree on how an edge is
/// divided. Doing it again changes nothing.
pub(super) fn share<K: Ord + Clone>(
    faces: &[[usize; 3]],
    pieces: &mut BTreeMap<usize, Piece<K>>,
    feature: impl Fn(&K) -> &Feature,
) {
    let mut around = FxHashMap::<[usize; 2], Vec<usize>>::default();
    for (index, nodes) in faces.iter().enumerate() {
        (0..3).for_each(|edge| {
            let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
            around.entry([a.min(b), a.max(b)]).or_default().push(index);
        })
    }
    let shared: Vec<(usize, K, Coordinate<3>)> = pieces
        .iter()
        .flat_map(|(&triangle, piece)| {
            let (around, feature) = (&around, &feature);
            piece.points.iter().flat_map(move |(key, point)| {
                let neighbours = match feature(key) {
                    Feature::Edge(edge) => around[edge].as_slice(),
                    _ => &[],
                };
                neighbours
                    .iter()
                    .filter(move |&&neighbour| neighbour != triangle)
                    .map(move |&neighbour| (neighbour, key.clone(), point.clone()))
            })
        })
        .collect();
    for (triangle, key, point) in shared {
        let piece = pieces.entry(triangle).or_default();
        if !piece.points.iter().any(|(known, _)| known == &key) {
            piece.points.push((key, point));
        }
    }
}
