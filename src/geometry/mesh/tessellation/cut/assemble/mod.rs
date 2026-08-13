#[cfg(test)]
mod test;

use super::{
    Class, EDGES, FACES, SNAP_QUALITY, Sign, Tables, Vertex,
    cleanup::{agglomerate, intern},
    face::{FaceCut, clip_face, face_cut},
    geometry::star_volume,
};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            quality::metrics::{Kind, minimum_scaled_jacobian},
            tessellation::{D, Tessellation},
        },
    },
    math::{CrossProduct, Quantity, Scalar, Tensor, TensorVec, unit::Length},
};
use std::{array::from_fn, collections::HashMap, collections::HashSet};

#[allow(clippy::type_complexity)]
fn build_cut_cells(
    mesh: &Mesh<D>,
    classes: &[Class],
    tables: &Tables,
    face_polygons: &HashMap<&[usize; 4], Vec<Vec<usize>>>,
    face_cuts: &HashMap<&[usize; 4], FaceCut>,
    crossing_ids: &HashMap<[usize; 2], Vec<usize>>,
    coordinates: &Coordinates<D>,
) -> Result<
    (
        Vec<[usize; 8]>,
        Vec<Vec<usize>>,
        Vec<[usize; 8]>,
        Vec<Vec<usize>>,
        Vec<usize>,
    ),
    &'static str,
> {
    let mut face_ids = HashMap::<Vec<usize>, usize>::new();
    let mut faces_nodes = Vec::new();
    let mut owners = Vec::new();
    let point = |vertex: Vertex| match vertex {
        Vertex::Node(node) => node,
        Vertex::Crossing(edge, ordinal) => crossing_ids[&edge][ordinal],
    };
    let mut hexes = Vec::new();
    let mut elements_faces = Vec::<Vec<usize>>::new();
    let mut sources = Vec::<[usize; 8]>::new();
    let mut offset = 0;
    mesh.iter().try_for_each(|block| {
        let local_faces = block.local_faces();
        block.iter().enumerate().try_for_each(|(local, element)| {
            match classes[offset + local] {
                Class::Inside => {
                    hexes.push(from_fn(|i| element[i]));
                    Ok(())
                }
                Class::Outside => Ok(()),
                Class::Cut => {
                    let interior = element
                        .iter()
                        .any(|node| tables.signs[node] == Sign::Inside);
                    let faces: Vec<([usize; 4], [usize; 4])> = local_faces
                        .iter()
                        .map(|face| {
                            let oriented = from_fn(|i| element[face[i]]);
                            let mut key = oriented;
                            key.sort_unstable();
                            (key, oriented)
                        })
                        .collect();
                    let mut adjacency = HashMap::<Vertex, Vec<Vertex>>::new();
                    faces.iter().for_each(|(key, _)| {
                        if let Some(pairs) = tables.segments.get(key) {
                            pairs.iter().for_each(|&[one, two]| {
                                adjacency.entry(one).or_default().push(two);
                                adjacency.entry(two).or_default().push(one);
                            })
                        }
                    });
                    EDGES.iter().for_each(|&[a, b]| {
                        let (na, nb) = (element[a], element[b]);
                        if tables.signs[&na] == Sign::On && tables.signs[&nb] == Sign::On {
                            let mut edge = [na, nb];
                            edge.sort_unstable();
                            let on_sides: Vec<Sign> = faces
                                .iter()
                                .filter(|(_, oriented)| {
                                    oriented.contains(&na) && oriented.contains(&nb)
                                })
                                .filter_map(|(key, _)| {
                                    face_cuts[key]
                                        .on_edges
                                        .iter()
                                        .find_map(|&(key, side)| (key == edge).then_some(side))
                                })
                                .collect();
                            if on_sides.contains(&Sign::Inside) && on_sides.contains(&Sign::Outside)
                            {
                                adjacency
                                    .entry(Vertex::Node(na))
                                    .or_default()
                                    .push(Vertex::Node(nb));
                                adjacency
                                    .entry(Vertex::Node(nb))
                                    .or_default()
                                    .push(Vertex::Node(na));
                            }
                        }
                    });
                    if adjacency.is_empty() {
                        if interior {
                            hexes.push(from_fn(|i| element[i]));
                        }
                        return Ok(());
                    }
                    if adjacency.values().any(|partners| partners.len() != 2) {
                        return Err("open cut chain within a cell");
                    }
                    let mut polygons = Vec::<Vec<usize>>::new();
                    faces.into_iter().try_for_each(|(key, oriented)| {
                        if face_cuts[&key].flush {
                            if interior {
                                return Err("refinement required at a face");
                            }
                        } else {
                            let corners = &tables.faces[&key];
                            let at = corners
                                .iter()
                                .position(|&node| node == oriented[0])
                                .unwrap();
                            let forward = corners[(at + 1) % 4] == oriented[1];
                            face_polygons[&key].iter().for_each(|polygon| {
                                polygons.push(if forward {
                                    polygon.clone()
                                } else {
                                    polygon.iter().rev().copied().collect()
                                })
                            })
                        }
                        Ok(())
                    })?;
                    let clipped = polygons.len();
                    let mut keys: Vec<Vertex> = adjacency.keys().copied().collect();
                    keys.sort_unstable();
                    let mut visited = HashSet::<Vertex>::new();
                    keys.into_iter().for_each(|start| {
                        if visited.insert(start) {
                            let mut polygon = vec![point(start)];
                            let mut previous = start;
                            let mut current = adjacency[&start][0];
                            while current != start {
                                visited.insert(current);
                                polygon.push(point(current));
                                let next = if adjacency[&current][0] == previous {
                                    adjacency[&current][1]
                                } else {
                                    adjacency[&current][0]
                                };
                                previous = current;
                                current = next;
                            }
                            if polygon.len() > 2 {
                                polygons.push(polygon);
                            }
                        }
                    });
                    if polygons.len() < 4 {
                        return Ok(());
                    }
                    let mut nodes: Vec<usize> = polygons.iter().flatten().copied().collect();
                    nodes.sort_unstable();
                    nodes.dedup();
                    let centroid = nodes
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .sum::<Coordinate<D>>()
                        / nodes.len() as Scalar;
                    polygons[clipped..].iter_mut().for_each(|polygon| {
                        let middle = &(polygon
                            .iter()
                            .map(|&node| coordinates[node].clone())
                            .sum::<Coordinate<D>>()
                            / polygon.len() as Scalar)
                            - &centroid;
                        let outward: Scalar = (0..polygon.len())
                            .map(|i| {
                                let one = &coordinates[polygon[i]] - &centroid;
                                let two =
                                    &coordinates[polygon[(i + 1) % polygon.len()]] - &centroid;
                                (one.cross(&two) * &middle).value()
                            })
                            .sum();
                        if outward < 0.0 {
                            polygon.reverse()
                        }
                    });
                    let mut roots = HashMap::new();
                    fn find(roots: &mut HashMap<usize, usize>, node: usize) -> usize {
                        let parent = *roots.entry(node).or_insert(node);
                        if parent == node {
                            node
                        } else {
                            let root = find(roots, parent);
                            roots.insert(node, root);
                            root
                        }
                    }
                    polygons.iter().for_each(|polygon| {
                        let root = find(&mut roots, polygon[0]);
                        polygon[1..].iter().for_each(|&node| {
                            let other = find(&mut roots, node);
                            roots.insert(other, root);
                        })
                    });
                    let components: HashSet<usize> = nodes
                        .into_iter()
                        .map(|node| find(&mut roots, node))
                        .collect();
                    if components.len() != 1 {
                        return Err("disconnected cell interior requires refinement");
                    }
                    let cell = elements_faces.len();
                    elements_faces.push(
                        polygons
                            .into_iter()
                            .map(|polygon| {
                                intern(&mut face_ids, &mut faces_nodes, &mut owners, polygon, cell)
                            })
                            .collect(),
                    );
                    sources.push(from_fn(|i| element[i]));
                    Ok(())
                }
            }
        })?;
        offset += block.number_of_elements();
        Ok(())
    })?;
    Ok((hexes, elements_faces, sources, faces_nodes, owners))
}

impl Tessellation {
    pub(super) fn assemble(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        tables: &Tables,
    ) -> Result<Mesh<D>, &'static str> {
        let mut coordinates = mesh.coordinates().clone();
        let mut crossing_ids = HashMap::<[usize; 2], Vec<usize>>::new();
        let mut crossing_edge = HashMap::<usize, [usize; 2]>::new();
        let mut crossed_edges: Vec<&[usize; 2]> = tables.crossings.keys().collect();
        crossed_edges.sort_unstable();
        crossed_edges.into_iter().for_each(|edge| {
            let ids: Vec<usize> = tables.crossings[edge]
                .iter()
                .map(|point| {
                    coordinates.push(point.clone());
                    let id = coordinates.len() - 1;
                    crossing_edge.insert(id, *edge);
                    id
                })
                .collect();
            crossing_ids.insert(*edge, ids);
        });
        let mut face_polygons = HashMap::new();
        let mut face_cuts = HashMap::new();
        tables.faces.iter().try_for_each(|(key, corners)| {
            let cut = face_cut(corners, &tables.signs, &tables.crossings)?;
            face_polygons.insert(
                key,
                if cut.flush {
                    Vec::new()
                } else {
                    clip_face(&cut, tables.segments.get(key), &crossing_ids)
                },
            );
            face_cuts.insert(key, cut);
            Ok(())
        })?;
        let (mut hexes, elements_faces, sources, mut faces_nodes, mut owners) = build_cut_cells(
            mesh,
            classes,
            tables,
            &face_polygons,
            &face_cuts,
            &crossing_ids,
            &coordinates,
        )?;
        let fractions: Vec<Scalar> = elements_faces
            .iter()
            .zip(sources.iter())
            .map(|(faces, hex)| {
                let polygons: Vec<Vec<usize>> = faces
                    .iter()
                    .map(|&face| faces_nodes[face].clone())
                    .collect();
                let reference: Vec<Vec<usize>> = FACES
                    .iter()
                    .map(|face| face.iter().map(|&local| hex[local]).collect())
                    .collect();
                star_volume(&polygons, &coordinates).ratio(star_volume(&reference, &coordinates))
            })
            .collect();
        let mut sets: Vec<HashSet<usize>> = elements_faces
            .iter()
            .map(|faces| faces.iter().copied().collect())
            .collect();
        let mut alive = agglomerate(
            &mut sets,
            &mut owners,
            &faces_nodes,
            &fractions,
            &coordinates,
        );
        let scales: Vec<Quantity<Length>> = sources
            .iter()
            .map(|hex| {
                EDGES
                    .iter()
                    .map(|&[a, b]| (&coordinates[hex[b]] - &coordinates[hex[a]]).norm())
                    .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
            })
            .collect();
        let whole: HashSet<usize> = hexes.iter().flatten().copied().collect();
        self.collapse_short_edges(
            &mut coordinates,
            &mut faces_nodes,
            &mut sets,
            &owners,
            &mut alive,
            &tables.signs,
            &whole,
            &scales,
            &crossing_edge,
        );
        (0..sets.len()).for_each(|cell| {
            if !alive[cell]
                || sets[cell].len() != 6
                || sets[cell].iter().any(|&face| faces_nodes[face].len() != 4)
            {
                return;
            }
            let mut uses = HashMap::new();
            sets[cell].iter().for_each(|&face| {
                faces_nodes[face]
                    .iter()
                    .for_each(|&node| *uses.entry(node).or_insert(0) += 1)
            });
            if uses.len() != 8 || uses.values().any(|&count| count != 3) {
                return;
            }
            let mut faces: Vec<usize> = sets[cell].iter().copied().collect();
            faces.sort_unstable();
            let bottom = faces[0];
            let mut base = faces_nodes[bottom].clone();
            if owners[bottom] == cell {
                base.reverse();
            }
            let vertical = |node: usize| -> Option<usize> {
                let mut counts = HashMap::new();
                faces[1..].iter().for_each(|&face| {
                    let polygon = &faces_nodes[face];
                    if let Some(at) = polygon.iter().position(|&other| other == node) {
                        [(at + 1) % 4, (at + 3) % 4].into_iter().for_each(|next| {
                            *counts.entry(polygon[next]).or_insert(0) += 1;
                        })
                    }
                });
                let partners: Vec<usize> = counts
                    .into_iter()
                    .filter(|&(partner, count)| count == 2 && !base.contains(&partner))
                    .map(|(partner, _)| partner)
                    .collect();
                (partners.len() == 1).then(|| partners[0])
            };
            let Some(top) = base
                .iter()
                .map(|&node| vertical(node))
                .collect::<Option<Vec<usize>>>()
            else {
                return;
            };
            let element: [usize; 8] = from_fn(|i| if i < 4 { base[i] } else { top[i - 4] });
            let mut expected: Vec<Vec<usize>> = FACES
                .iter()
                .map(|face| {
                    let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                    key.sort_unstable();
                    key
                })
                .collect();
            expected.sort_unstable();
            let mut actual: Vec<Vec<usize>> = faces
                .iter()
                .map(|&face| {
                    let mut key = faces_nodes[face].clone();
                    key.sort_unstable();
                    key
                })
                .collect();
            actual.sort_unstable();
            if expected == actual
                && minimum_scaled_jacobian(Kind::Hexahedron, &element, &coordinates) >= SNAP_QUALITY
            {
                hexes.push(element);
                alive[cell] = false;
            }
        });
        let mut compacted = HashMap::new();
        let mut compact_nodes = Vec::new();
        let elements_faces: Vec<Vec<usize>> = alive
            .into_iter()
            .zip(sets)
            .enumerate()
            .filter_map(|(cell, (kept, set))| {
                kept.then(|| {
                    let mut faces: Vec<usize> = set.into_iter().collect();
                    faces.sort_unstable();
                    faces
                        .into_iter()
                        .map(|face| {
                            *compacted.entry(face).or_insert_with(|| {
                                let mut polygon = faces_nodes[face].clone();
                                if owners[face] != cell {
                                    polygon.reverse()
                                }
                                compact_nodes.push(polygon);
                                compact_nodes.len() - 1
                            })
                        })
                        .collect()
                })
            })
            .collect();
        let faces_nodes = compact_nodes;
        let mut remap = HashMap::new();
        let mut points = Coordinates::new();
        let mut renumber = |node: usize, points: &mut Coordinates<D>| -> usize {
            *remap.entry(node).or_insert_with(|| {
                points.push(coordinates[node].clone());
                points.len() - 1
            })
        };
        let hexes: Vec<[usize; 8]> = hexes
            .into_iter()
            .map(|hex| hex.map(|node| renumber(node, &mut points)))
            .collect();
        let faces_nodes: Vec<Vec<usize>> = faces_nodes
            .into_iter()
            .map(|face| {
                face.into_iter()
                    .map(|node| renumber(node, &mut points))
                    .collect()
            })
            .collect();
        let mut connectivities = Vec::new();
        if !hexes.is_empty() {
            connectivities.push(Connectivity::Hexahedral(hexes.into()));
        }
        if !elements_faces.is_empty() {
            connectivities.push(Connectivity::Polyhedral(
                (elements_faces, faces_nodes).into(),
            ));
        }
        Ok((connectivities, points).into())
    }
}
