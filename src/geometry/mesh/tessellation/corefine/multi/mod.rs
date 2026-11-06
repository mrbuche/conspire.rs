//! The surfaces between any number of solids, cut along each other at once.

mod junction;
mod names;
#[cfg(test)]
mod test;

use self::{
    junction::{Ends, Located, arrange, edge_junctions, junctions, planar_junctions},
    names::{Groups, Name, Offsets, Spot, solid_of, viewed},
};
use super::{
    Contacts, Feature, Key, Region, Side,
    classify::{coplanar, patches, relate},
    frame::{coincident, cut_order},
    split::{Piece, refine, share},
};
use crate::math::{FxHashMap, FxHashSet};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Tessellation, Watertight},
        predicates::Cutter,
    },
    math::Tensor,
};
use alloc::collections::{BTreeMap, BTreeSet};
use core::cmp::Ordering;

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
    /// different solids meet at shared vertices. Any number of surfaces may meet at a point inside
    /// a triangle of each, and any number of such points may be on one curve. It fails where they
    /// meet on an edge or a vertex of a triangle, which it cannot yet cut at, and wherever the
    /// surfaces of two solids cannot be cut.
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
    let mut renumbered = FxHashMap::<usize, usize>::default();
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

/// Names the points that are the same point by the same name.
///
/// Two points on an edge that what cuts the edge puts at one place are one point, though the
/// surfaces that meet there were compared two at a time, so each is named by only two of them. The
/// name of both is then the lowest feature of every surface the two are on, which is also what the
/// names of the same point in the triangles of the other solids become.
fn unify<'a>(
    pieces: &mut [BTreeMap<usize, Piece<Spot>>],
    faces: &[Vec<[usize; 3]>],
    coordinates: &[Coordinate<3>],
    offsets: &[Offsets],
    links: &[[Name; 2]],
    cutter: impl Fn(&Feature) -> Cutter<'a>,
) -> Result<(), &'static str> {
    for round in 0..4 {
        pieces
            .iter_mut()
            .enumerate()
            .for_each(|(volume, map)| share(&faces[volume], map, |spot| &spot.own));
        let mut groups = Groups::<Name>::default();
        links.iter().for_each(|[a, b]| groups.join(a, b));
        let mut found = false;
        for (volume, map) in pieces.iter().enumerate() {
            for (&triangle, piece) in map {
                let nodes = faces[volume][triangle];
                let same = coincident(
                    nodes.map(|node| &coordinates[node]),
                    nodes,
                    &piece.points,
                    |spot| &spot.own,
                    |spot| spot.others.iter().map(&cutter).collect(),
                )?;
                for [a, b] in same {
                    groups.join(&Name::from(&a), &Name::from(&b));
                    found = true;
                }
            }
        }
        if !found && (round > 0 || links.is_empty()) {
            return Ok(());
        }
        let mut merged = FxHashMap::<Name, BTreeSet<Feature>>::default();
        for piece in pieces.iter().flat_map(|map| map.values()) {
            for (spot, _) in &piece.points {
                let name = Name::from(spot);
                let root = groups.find(&name);
                merged
                    .entry(root)
                    .or_default()
                    .extend(name.features().iter().cloned());
            }
        }
        let mut renamed = |spot: &Spot, volume: usize| {
            let features = &merged[&groups.find(&Name::from(spot))];
            let mut lowest = BTreeMap::<usize, Feature>::new();
            for feature in features {
                let entry = lowest
                    .entry(solid_of(feature, offsets))
                    .or_insert_with(|| feature.clone());
                *entry = (*entry).clone().min(feature.clone());
            }
            let own = lowest.remove(&volume).unwrap_or_else(|| spot.own.clone());
            let mut others: Vec<Feature> = lowest.into_values().collect();
            others.sort_unstable();
            Spot { own, others }
        };
        for (volume, map) in pieces.iter_mut().enumerate() {
            for piece in map.values_mut() {
                for (spot, _) in &mut piece.points {
                    *spot = renamed(spot, volume);
                }
                for segment in &mut piece.segments {
                    *segment = [renamed(&segment[0], volume), renamed(&segment[1], volume)];
                }
                piece.segments.retain(|[a, b]| a != b);
            }
        }
    }
    Err("the points where surfaces meet cannot be named")
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
    let pairs: BTreeMap<[usize; 2], FxHashSet<[usize; 2]>> = contacts
        .iter()
        .map(|(&pair, contacts)| {
            let crossing = contacts.crossings.iter();
            (pair, crossing.map(|c| [c.first, c.second]).collect())
        })
        .collect();
    let corners = |solid: usize, triangle: usize| {
        triangles[offsets[solid].triangle + triangle].map(|node| &coordinates[node])
    };
    let cutter = |feature: &Feature| match feature {
        Feature::Face(triangle) => {
            Cutter::Face(triangles[*triangle].map(|node| &coordinates[node]))
        }
        Feature::Edge([a, b]) => Cutter::Edge([&coordinates[*a], &coordinates[*b]]),
        Feature::Vertex(node) => Cutter::Vertex(&coordinates[*node]),
    };
    let flat: BTreeMap<[usize; 2], Vec<[usize; 2]>> = contacts
        .iter()
        .map(|(&pair, contacts)| (pair, contacts.coplanar.clone()))
        .collect();
    let mut junctions = junctions(&pairs, solids.len(), corners)?;
    junctions.extend(edge_junctions(&flat, &pairs, solids.len(), corners)?);
    let counts: Vec<usize> = faces.iter().map(Vec::len).collect();
    junctions.extend(planar_junctions(&flat, &counts, corners)?);
    let curves: FxHashMap<_, _> = contacts
        .iter()
        .flat_map(|(&[i, j], contacts)| {
            let sides = [&offsets[i], &offsets[j]];
            contacts.crossings.iter().map(move |c| {
                let features = c.keys.each_ref().map(|key| {
                    let key = viewed(key, true, sides);
                    [key.first, key.second]
                });
                let ends = Ends {
                    points: c.segment.clone(),
                    features,
                };
                (([i, j], c.first, c.second), ends)
            })
        })
        .collect();
    let feature = |solid: usize, triangle: usize, located: Located| {
        let nodes = triangles[offsets[solid].triangle + triangle];
        match located {
            Located::Face => Feature::Face(offsets[solid].triangle + triangle),
            Located::Edge(edge) => {
                let (a, b) = (nodes[edge], nodes[(edge + 1) % 3]);
                Feature::Edge([a.min(b), a.max(b)])
            }
            Located::Vertex(corner) => Feature::Vertex(nodes[corner]),
        }
    };
    let (points, through, ends) = arrange(&junctions, &curves, feature, corners)?;
    let features: Vec<Vec<Feature>> = points
        .iter()
        .map(|point| {
            point
                .places
                .iter()
                .map(|&(solid, triangle, located)| feature(solid, triangle, located))
                .collect()
        })
        .collect();
    let links: Vec<[Name; 2]> = ends
        .iter()
        .map(|(point, ends)| {
            [
                Name::new(ends.iter().cloned()),
                Name::new(features[*point].iter().cloned()),
            ]
        })
        .collect();
    let mut by_edge = FxHashMap::<Feature, Vec<usize>>::default();
    for (point, found) in features.iter().enumerate() {
        for feature in found.iter().filter(|f| matches!(f, Feature::Edge(_))) {
            by_edge.entry(feature.clone()).or_default().push(point);
        }
    }
    // The points inside a triangle that are on an edge of the other solid, in the plane of the
    // triangle, that a segment of the edge crosses the triangle along, from its start to its end.
    let on_constraint = |constraint: &crate::geometry::mesh::Constraint, sides: [&Offsets; 2]| {
        let (volume, cut, cutter_side) = match constraint.side {
            Side::First => (0, 0, 1),
            Side::Second => (1, 1, 0),
        };
        let ends = constraint.keys.each_ref().map(|key| {
            let key = viewed(key, true, sides);
            [key.first, key.second]
        });
        let cutter_features = ends.each_ref().map(|end| &end[cutter_side]);
        let edge = cutter_features
            .iter()
            .find(|feature| matches!(feature, Feature::Edge(_)))
            .map(|feature| (*feature).clone())
            .or_else(|| match cutter_features {
                [Feature::Vertex(a), Feature::Vertex(b)] => {
                    Some(Feature::Edge([*a.min(b), *a.max(b)]))
                }
                _ => None,
            });
        let Some(edge @ Feature::Edge([n0, n1])) = edge else {
            return Ok(Vec::new());
        };
        let _ = (volume, cut);
        let triangle = Feature::Face(sides[volume].triangle + constraint.triangle);
        let mut found: Vec<usize> = by_edge
            .get(&edge)
            .into_iter()
            .flatten()
            .copied()
            .filter(|&point| features[point].contains(&triangle))
            .collect();
        if found.is_empty() {
            return Ok(found);
        }
        let (p, q) = (&coordinates[n0], &coordinates[n1]);
        let cutters = |features: Vec<&Feature>| -> Vec<Cutter> {
            features
                .into_iter()
                .filter(|feature| **feature != edge && **feature != triangle)
                .map(&cutter)
                .collect()
        };
        let [start, finish] = ends.each_ref().map(|end| cutters(end.iter().collect()));
        let forward = cut_order(p, q, &start, &finish)?;
        let at = |point: usize| cutters(features[point].iter().collect());
        let mut between = Vec::new();
        for point in found.drain(..) {
            let here = at(point);
            let before = cut_order(p, q, &start, &here)?;
            let after = cut_order(p, q, &here, &finish)?;
            if before == forward && after == forward {
                between.push(point);
            }
        }
        for i in 1..between.len() {
            let mut j = i;
            while j > 0 {
                match cut_order(p, q, &at(between[j - 1]), &at(between[j]))? {
                    o if o == forward => break,
                    Ordering::Equal => {
                        eprintln!(
                            "SAMEPTS {edge:?} {:?} {:?} {:?}",
                            features[between[j - 1]],
                            features[between[j]],
                            triangle
                        );
                        return Err("two triple points are the same point");
                    }
                    _ => between.swap(j - 1, j),
                }
                j -= 1;
            }
        }
        Ok(between)
    };
    let spot = |point: usize, volume: usize| {
        let own = points[point]
            .places
            .iter()
            .position(|&(solid, ..)| solid == volume)
            .expect("a point is on a curve of a solid it is not in");
        let mut others: Vec<Feature> = features[point]
            .iter()
            .enumerate()
            .filter(|&(n, _)| n != own)
            .map(|(_, feature)| feature.clone())
            .collect();
        others.sort_unstable();
        Spot {
            own: features[point][own].clone(),
            others,
        }
    };
    let mut pieces: Vec<BTreeMap<usize, Piece<Spot>>> =
        solids.iter().map(|_| BTreeMap::new()).collect();
    let mut identical = Groups::<usize>::default();
    for (&[i, j], contacts) in &contacts {
        let sides = [&offsets[i], &offsets[j]];
        let mut add = |side: Side,
                       triangle: usize,
                       keys: &[Key; 2],
                       segment: &[Coordinate<3>; 2],
                       split: &[usize]| {
            let (volume, from_first) = match side {
                Side::First => (i, true),
                Side::Second => (j, false),
            };
            let [start, end] = keys
                .each_ref()
                .map(|key| Spot::from(viewed(key, from_first, sides)));
            let piece = pieces[volume].entry(triangle).or_default();
            let mut keys = vec![start];
            let mut positions = vec![segment[0].clone()];
            for &point in split {
                keys.push(spot(point, volume));
                positions.push(points[point].position.clone());
            }
            keys.push(end);
            positions.push(segment[1].clone());
            for (keys, ends) in keys.windows(2).zip(positions.windows(2)) {
                piece.add(
                    &[keys[0].clone(), keys[1].clone()],
                    &[ends[0].clone(), ends[1].clone()],
                );
            }
        };
        for crossing in &contacts.crossings {
            let split = through
                .get(&([i, j], crossing.first, crossing.second))
                .map_or(&[][..], |found| found.as_slice());
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
            let split = on_constraint(constraint, sides)?;
            add(
                constraint.side,
                constraint.triangle,
                &constraint.keys,
                &constraint.segment,
                &split,
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
    unify(&mut pieces, &faces, &coordinates, &offsets, &links, cutter)?;
    let mut vertices = coordinates.clone();
    let mut named = FxHashMap::<Name, usize>::default();
    let mut refined = Vec::new();
    for (volume, own) in faces.iter().enumerate() {
        refined.push(refine(
            own,
            |node| &coordinates[node],
            core::mem::take(&mut pieces[volume]),
            |spot| &spot.own,
            |spot| spot.others.iter().map(cutter).collect(),
            |spot, position| {
                let nodes: Vec<usize> = spot
                    .others
                    .iter()
                    .filter_map(|feature| match feature {
                        Feature::Vertex(node) => Some(*node),
                        _ => None,
                    })
                    .collect();
                match nodes[..] {
                    [first, ..] => {
                        nodes.iter().for_each(|node| identical.join(&first, node));
                        first
                    }
                    [] => *named.entry(Name::from(spot)).or_insert_with(|| {
                        vertices.push(position.clone());
                        vertices.len() - 1
                    }),
                }
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
            .collect::<FxHashSet<_>>();
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
