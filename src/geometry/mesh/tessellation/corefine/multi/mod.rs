//! The surfaces between any number of solids, cut along each other at once.

#[cfg(test)]
mod test;

use super::{
    Contacts, Feature, Key, Region, Side,
    classify::{coplanar, patches, relate},
    split::{Piece, refine},
};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Tessellation, Watertight},
        predicates::Cutter,
    },
    math::Tensor,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// The surfaces that separate any number of solids, each triangle labelled with the solid on
/// either side of it.
///
/// A solid is a material, and where solids overlap the later one owns the space, as if each were
/// drawn over the ones before it. A triangle is kept where it separates two materials, or a
/// material from nothing, so a surface that ends up inside a later solid is not kept, and where
/// surfaces coincide one triangle is kept, not two.
pub struct Interfaces {
    surface: Tessellation,
    /// The material on the inner side of each triangle of the surface, the side its normal points
    /// away from.
    inside: Vec<usize>,
    /// The material on the outer side of each triangle, or `None` where there is none.
    outside: Vec<Option<usize>>,
}

impl Interfaces {
    /// The surfaces between the solids, in the order they are drawn.
    ///
    /// The surfaces are cut along each other at once, from the original triangles, so triangles of
    /// different solids meet at shared vertices. It fails where three surfaces meet at a point,
    /// which it cannot yet cut at, and wherever the surfaces of two solids cannot be cut.
    pub fn new(solids: &[Watertight]) -> Result<Self, &'static str> {
        build(solids)
    }
    /// The triangles of every interface, as one surface.
    pub fn surface(&self) -> &Tessellation {
        &self.surface
    }
    /// The material on the inner side of each triangle, the side its normal points away from.
    pub fn inside(&self) -> &[usize] {
        &self.inside
    }
    /// The material on the outer side of each triangle, or `None` where there is none.
    pub fn outside(&self) -> &[Option<usize>] {
        &self.outside
    }
    /// The closed surface around a material: its triangles as they are, and those of the
    /// materials around it turned over.
    pub fn boundary(&self, material: usize) -> Result<Tessellation, &'static str> {
        let coordinates = self.surface.mesh().coordinates();
        let mut faces = Vec::new();
        for ((nodes, &inside), &outside) in self
            .surface
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .zip(&self.inside)
            .zip(&self.outside)
        {
            if inside == material {
                faces.push([nodes[0], nodes[1], nodes[2]]);
            } else if outside == Some(material) {
                faces.push([nodes[0], nodes[2], nodes[1]]);
            }
        }
        if faces.is_empty() {
            return Err("the material has no surface");
        }
        let positions: Vec<&Coordinate<3>> = coordinates.iter().collect();
        Ok(surface(faces, &positions))
    }
}

/// A surface of the faces that use the `positions`, without the positions nothing uses.
fn surface(faces: Vec<[usize; 3]>, positions: &[&Coordinate<3>]) -> Tessellation {
    let mut renumbered: HashMap<usize, usize> = HashMap::new();
    let mut vertices: Vec<Coordinate<3>> = Vec::new();
    let faces: Vec<[usize; 3]> = faces
        .into_iter()
        .map(|face| {
            face.map(|node| {
                *renumbered.entry(node).or_insert_with(|| {
                    vertices.push(positions[node].clone());
                    vertices.len() - 1
                })
            })
        })
        .collect();
    (faces, vertices).into()
}

/// Where a solid's numbers start in those of all the solids together.
struct Offsets {
    node: usize,
    triangle: usize,
}

fn globalize(feature: &Feature, offsets: &Offsets) -> Feature {
    match feature {
        Feature::Vertex(node) => Feature::Vertex(node + offsets.node),
        Feature::Edge([a, b]) => Feature::Edge([a + offsets.node, b + offsets.node]),
        Feature::Face(triangle) => Feature::Face(triangle + offsets.triangle),
    }
}

/// The key of a point as seen from one of the two surfaces it is on, with that surface's feature
/// first and the other's second, named by the numbers of all the solids together.
fn viewed(key: &Key, from_first: bool, offsets: [&Offsets; 2]) -> Key {
    let (first, second) = (
        globalize(&key.first, offsets[0]),
        globalize(&key.second, offsets[1]),
    );
    if from_first {
        Key { first, second }
    } else {
        Key {
            first: second,
            second: first,
        }
    }
}

/// The same point seen from the other surface is the same point.
fn canonical(key: &Key) -> Key {
    if key.first <= key.second {
        key.clone()
    } else {
        Key {
            first: key.second.clone(),
            second: key.first.clone(),
        }
    }
}

fn find(parents: &mut HashMap<usize, usize>, node: usize) -> usize {
    let parent = *parents.entry(node).or_insert(node);
    if parent == node {
        node
    } else {
        let root = find(parents, parent);
        parents.insert(node, root);
        root
    }
}

fn build(solids: &[Watertight]) -> Result<Interfaces, &'static str> {
    let tessellations: Vec<&Tessellation> = solids.iter().map(Watertight::tessellation).collect();
    let mut offsets: Vec<Offsets> = Vec::new();
    let mut coordinates: Vec<Coordinate<3>> = Vec::new();
    let mut triangles: Vec<[usize; 3]> = Vec::new();
    let mut faces: Vec<Vec<[usize; 3]>> = Vec::new();
    for tessellation in &tessellations {
        let node = coordinates.len();
        offsets.push(Offsets {
            node,
            triangle: triangles.len(),
        });
        coordinates.extend(tessellation.mesh().coordinates().iter().cloned());
        let own: Vec<[usize; 3]> = tessellation
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .map(|nodes| [nodes[0] + node, nodes[1] + node, nodes[2] + node])
            .collect();
        triangles.extend(own.iter().copied());
        faces.push(own);
    }
    let mut contacts: BTreeMap<[usize; 2], Contacts> = BTreeMap::new();
    for i in 0..solids.len() {
        for j in i + 1..solids.len() {
            contacts.insert([i, j], tessellations[i].contacts(tessellations[j])?);
        }
    }
    let mut pieces: Vec<BTreeMap<usize, Piece>> = solids.iter().map(|_| BTreeMap::new()).collect();
    let mut identical: HashMap<usize, usize> = HashMap::new();
    for (&[i, j], contacts) in &contacts {
        let sides = [&offsets[i], &offsets[j]];
        let mut add =
            |side: Side, triangle: usize, keys: &[Key; 2], segment: &[Coordinate<3>; 2]| {
                let (volume, from_first) = match side {
                    Side::First => (i, true),
                    Side::Second => (j, false),
                };
                let viewed = keys.each_ref().map(|key| viewed(key, from_first, sides));
                pieces[volume]
                    .entry(triangle)
                    .or_default()
                    .add(&viewed, segment);
            };
        for crossing in &contacts.crossings {
            add(
                Side::First,
                crossing.first,
                &crossing.keys,
                &crossing.segment,
            );
            add(
                Side::Second,
                crossing.second,
                &crossing.keys,
                &crossing.segment,
            );
        }
        for constraint in &contacts.constraints {
            add(
                constraint.side,
                constraint.triangle,
                &constraint.keys,
                &constraint.segment,
            );
        }
        let keys = contacts
            .crossings
            .iter()
            .flat_map(|crossing| &crossing.keys)
            .chain(contacts.constraints.iter().flat_map(|c| &c.keys));
        for key in keys {
            if let (Feature::Vertex(a), Feature::Vertex(b)) = (&key.first, &key.second) {
                let (a, b) = (a + sides[0].node, b + sides[1].node);
                let (a, b) = (find(&mut identical, a), find(&mut identical, b));
                identical.insert(a.max(b), a.min(b));
            }
        }
    }
    let cutter = |feature: &Feature| match feature {
        Feature::Face(triangle) => {
            Cutter::Face(triangles[*triangle].map(|node| &coordinates[node]))
        }
        Feature::Edge([a, b]) => Cutter::Edge([&coordinates[*a], &coordinates[*b]]),
        Feature::Vertex(node) => Cutter::Vertex(&coordinates[*node]),
    };
    let mut vertices = coordinates.clone();
    let mut named: HashMap<Key, usize> = HashMap::new();
    let mut refined = Vec::new();
    for (volume, own) in faces.iter().enumerate() {
        refined.push(refine(
            own,
            |node| &coordinates[node],
            std::mem::take(&mut pieces[volume]),
            Side::First,
            cutter,
            |key, position| match &key.second {
                Feature::Vertex(node) => *node,
                _ => *named.entry(canonical(key)).or_insert_with(|| {
                    vertices.push(position.clone());
                    vertices.len() - 1
                }),
            },
        )?);
    }
    let mut representative = HashMap::new();
    for &node in identical.clone().keys() {
        representative.insert(node, find(&mut identical, node));
    }
    let renumber = |node: usize| representative.get(&node).copied().unwrap_or(node);
    for cut in &mut refined {
        cut.faces
            .iter_mut()
            .for_each(|face| *face = face.map(renumber));
        cut.curves = cut
            .curves
            .iter()
            .map(|&[a, b]| {
                let (a, b) = (renumber(a), renumber(b));
                [a.min(b), a.max(b)]
            })
            .collect::<HashSet<_>>();
    }
    let positions: Vec<&Coordinate<3>> = vertices.iter().collect();
    let mut kept: Vec<[usize; 3]> = Vec::new();
    let mut inside = Vec::new();
    let mut outside = Vec::new();
    for (volume, cut) in refined.iter().enumerate() {
        let groups = patches(&cut.faces, &cut.curves);
        let mut regions: Vec<(usize, Vec<Region>)> = Vec::new();
        for other in (0..solids.len()).filter(|&other| other != volume) {
            let (pair, side) = if volume < other {
                ([volume, other], Side::First)
            } else {
                ([other, volume], Side::Second)
            };
            let relations = relate(
                &cut.faces,
                |node| positions[node],
                &cut.parents,
                &groups,
                tessellations[other],
                &solids[other],
                &coplanar(&contacts[&pair], side),
            )?;
            regions.push((other, relations));
        }
        for (face, nodes) in cut.faces.iter().enumerate() {
            let relations = regions
                .iter()
                .map(|(other, found)| (*other, found[face]))
                .collect::<Vec<_>>();
            let mut within = vec![volume];
            let mut beyond = Vec::new();
            let mut doubled = false;
            for &(other, region) in &relations {
                match region {
                    Region::Inside => {
                        within.push(other);
                        beyond.push(other);
                    }
                    Region::Coincident { aligned: true } => within.push(other),
                    Region::Coincident { aligned: false } => {
                        beyond.push(other);
                        doubled |= other > volume;
                    }
                    Region::Outside => (),
                }
            }
            if within.iter().max() == Some(&volume) && !doubled {
                kept.push(*nodes);
                inside.push(volume);
                outside.push(beyond.into_iter().max());
            }
        }
    }
    if kept.is_empty() {
        return Err("there is no surface between the solids");
    }
    Ok(Interfaces {
        surface: surface(kept, &positions),
        inside,
        outside,
    })
}
