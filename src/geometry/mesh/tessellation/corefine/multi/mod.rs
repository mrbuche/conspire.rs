//! The surfaces between any number of solids, cut along each other at once.

mod junction;
mod names;
#[cfg(test)]
mod test;

use self::{
    junction::junctions,
    names::{Groups, Name, Offsets, Spot, viewed},
};
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
    /// different solids meet at shared vertices. Three surfaces may meet at a point inside a
    /// triangle of each. It fails where they meet on an edge or a vertex, or where several such
    /// points are on one curve, which it cannot yet cut at, and wherever the surfaces of two
    /// solids cannot be cut.
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
    let mut renumbered = HashMap::<usize, usize>::new();
    let mut vertices = Vec::<Coordinate<3>>::new();
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

fn build(solids: &[Watertight]) -> Result<Interfaces, &'static str> {
    let tessellations: Vec<&Tessellation> = solids.iter().map(Watertight::tessellation).collect();
    let mut offsets = Vec::<Offsets>::new();
    let mut coordinates = Vec::<Coordinate<3>>::new();
    let mut triangles = Vec::<[usize; 3]>::new();
    let mut faces = Vec::<Vec<[usize; 3]>>::new();
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
    let pairs: BTreeMap<[usize; 2], HashSet<[usize; 2]>> = contacts
        .iter()
        .map(|(&pair, contacts)| {
            let crossing = contacts.crossings.iter();
            (pair, crossing.map(|c| [c.first, c.second]).collect())
        })
        .collect();
    let junctions = junctions(&pairs, solids.len(), |solid, triangle| {
        triangles[offsets[solid].triangle + triangle].map(|node| &coordinates[node])
    })?;
    let mut through = HashMap::<([usize; 2], usize, usize), Vec<usize>>::new();
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
    let mut pieces: Vec<BTreeMap<usize, Piece<Spot>>> =
        solids.iter().map(|_| BTreeMap::new()).collect();
    let mut identical = Groups::<usize>::default();
    for (&[i, j], contacts) in &contacts {
        let sides = [&offsets[i], &offsets[j]];
        let mut add = |side: Side,
                       triangle: usize,
                       keys: &[Key; 2],
                       segment: &[Coordinate<3>; 2],
                       split: Option<usize>| {
            let (volume, from_first) = match side {
                Side::First => (i, true),
                Side::Second => (j, false),
            };
            let [start, end] = keys
                .each_ref()
                .map(|key| Spot::from(viewed(key, from_first, sides)));
            let piece = pieces[volume].entry(triangle).or_default();
            let Some(junction) = split.map(|index| &junctions[index]) else {
                piece.add(&[start, end], segment);
                return;
            };
            let others: Vec<Feature> = junction
                .triangles
                .iter()
                .filter(|&&(solid, _)| solid != volume)
                .map(|&(solid, triangle)| Feature::Face(triangle + offsets[solid].triangle))
                .collect();
            let middle = Spot {
                own: Feature::Face(triangle + offsets[volume].triangle),
                others,
            };
            let position = &junction.position;
            piece.add(
                &[start, middle.clone()],
                &[segment[0].clone(), position.clone()],
            );
            piece.add(&[middle, end], &[position.clone(), segment[1].clone()]);
        };
        for crossing in &contacts.crossings {
            let split = match through.get(&([i, j], crossing.first, crossing.second)) {
                None => None,
                Some(found) if found.len() == 1 => Some(found[0]),
                Some(_) => return Err("several triple points on one curve"),
            };
            add(
                Side::First,
                crossing.first,
                &crossing.keys,
                &crossing.segment,
                split,
            );
            add(
                Side::Second,
                crossing.second,
                &crossing.keys,
                &crossing.segment,
                split,
            );
        }
        for constraint in &contacts.constraints {
            add(
                constraint.side,
                constraint.triangle,
                &constraint.keys,
                &constraint.segment,
                None,
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
                identical.join(&a, &b);
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
    let mut named = HashMap::<Name, usize>::new();
    let mut refined = Vec::new();
    for (volume, own) in faces.iter().enumerate() {
        refined.push(refine(
            own,
            |node| &coordinates[node],
            std::mem::take(&mut pieces[volume]),
            |spot| &spot.own,
            |spot| spot.others.iter().map(cutter).collect(),
            |spot, position| match spot.others[..] {
                [Feature::Vertex(node)] => node,
                _ => *named.entry(Name::from(spot)).or_insert_with(|| {
                    vertices.push(position.clone());
                    vertices.len() - 1
                }),
            },
        )?);
    }
    let representative = identical.roots();
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
