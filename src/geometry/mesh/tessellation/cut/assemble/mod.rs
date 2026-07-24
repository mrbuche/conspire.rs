#[cfg(test)]
mod test;

use super::{
    COLLAPSE_FRACTION, Class, EDGES, FACES, SLIVER_FRACTION, SNAP_QUALITY, Sign, Tables, Vertex,
    face::{FaceCut, clip_face, face_cut},
    geometry::{face_area, signed_volume, star_volume},
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
    math::{CrossProduct, Scalar, Tensor, TensorVec},
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
    fn intern(
        face_ids: &mut HashMap<Vec<usize>, usize>,
        faces_nodes: &mut Vec<Vec<usize>>,
        owners: &mut Vec<usize>,
        polygon: Vec<usize>,
        cell: usize,
    ) -> usize {
        let mut key = polygon.clone();
        key.sort_unstable();
        *face_ids.entry(key).or_insert_with(|| {
            faces_nodes.push(polygon);
            owners.push(cell);
            faces_nodes.len() - 1
        })
    }
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
                    if polygons.is_empty() {
                        return Ok(());
                    }
                    let nodes: HashSet<usize> = polygons.iter().flatten().copied().collect();
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
                                one.cross(&two) * &middle
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
                star_volume(&polygons, &coordinates) / star_volume(&reference, &coordinates)
            })
            .collect();
        let mut sets: Vec<HashSet<usize>> = elements_faces
            .iter()
            .map(|faces| faces.iter().copied().collect())
            .collect();
        let mut face_polys = HashMap::<usize, Vec<usize>>::new();
        sets.iter().enumerate().for_each(|(cell, faces)| {
            faces
                .iter()
                .for_each(|&face| face_polys.entry(face).or_default().push(cell))
        });
        let mut alive = vec![true; sets.len()];
        let mut slivers: Vec<usize> = (0..sets.len())
            .filter(|&cell| fractions[cell] < SLIVER_FRACTION)
            .collect();
        slivers.sort_by(|&one, &two| {
            fractions[one]
                .partial_cmp(&fractions[two])
                .unwrap()
                .then(one.cmp(&two))
        });
        slivers.into_iter().for_each(|sliver| {
            if !alive[sliver] {
                return;
            }
            let mut areas = HashMap::new();
            sets[sliver].iter().for_each(|&face| {
                face_polys[&face].iter().for_each(|&other| {
                    if other != sliver && alive[other] {
                        *areas.entry(other).or_insert(0.0) +=
                            face_area(&faces_nodes[face], &coordinates);
                    }
                })
            });
            if let Some(target) = areas
                .into_iter()
                .max_by(|(one, area_one), (two, area_two)| {
                    area_one.partial_cmp(area_two).unwrap().then(two.cmp(one))
                })
                .map(|(other, _)| other)
            {
                let common: Vec<usize> = sets[sliver]
                    .iter()
                    .copied()
                    .filter(|face| sets[target].contains(face))
                    .collect();
                let moved: Vec<usize> = sets[sliver]
                    .iter()
                    .copied()
                    .filter(|face| !common.contains(face))
                    .collect();
                common.iter().for_each(|face| {
                    sets[target].remove(face);
                });
                moved.into_iter().for_each(|face| {
                    if owners[face] == sliver {
                        owners[face] = target
                    }
                    sets[target].insert(face);
                    face_polys
                        .get_mut(&face)
                        .unwrap()
                        .iter_mut()
                        .for_each(|cell| {
                            if *cell == sliver {
                                *cell = target
                            }
                        });
                });
                alive[sliver] = false;
            }
        });
        self.collapse_short_edges(
            &mut coordinates,
            &mut faces_nodes,
            &mut sets,
            &owners,
            &mut alive,
            tables,
            &hexes,
            &sources,
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
    #[allow(clippy::too_many_arguments)]
    fn collapse_short_edges(
        &self,
        coordinates: &mut Coordinates<D>,
        faces_nodes: &mut [Vec<usize>],
        sets: &mut [HashSet<usize>],
        owners: &[usize],
        alive: &mut [bool],
        tables: &Tables,
        hexes: &[[usize; 8]],
        sources: &[[usize; 8]],
        crossing_edge: &HashMap<usize, [usize; 2]>,
    ) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let surface_elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let mut face_cells = HashMap::<usize, Vec<usize>>::new();
        sets.iter().enumerate().for_each(|(cell, faces)| {
            if alive[cell] {
                faces
                    .iter()
                    .for_each(|&face| face_cells.entry(face).or_default().push(cell))
            }
        });
        let mut node_faces = HashMap::<usize, HashSet<usize>>::new();
        face_cells.keys().for_each(|&face| {
            faces_nodes[face].iter().for_each(|&node| {
                node_faces.entry(node).or_default().insert(face);
            })
        });
        let scales: Vec<Scalar> = sources
            .iter()
            .map(|hex| {
                EDGES
                    .iter()
                    .map(|&[a, b]| (&coordinates[hex[b]] - &coordinates[hex[a]]).norm())
                    .fold(Scalar::INFINITY, Scalar::min)
            })
            .collect();
        let mut ranks = HashMap::new();
        tables.signs.iter().for_each(|(&node, &sign)| {
            ranks.insert(node, if sign == Sign::On { 2 } else { 1 });
        });
        hexes.iter().flatten().for_each(|&node| {
            ranks.insert(node, 3);
        });
        let rank = |node: usize| ranks.get(&node).copied().unwrap_or(0);
        let mut short = Vec::new();
        face_cells.iter().for_each(|(&face, cells)| {
            let polygon = &faces_nodes[face];
            let limit = COLLAPSE_FRACTION
                * cells
                    .iter()
                    .map(|&cell| scales[cell])
                    .fold(Scalar::INFINITY, Scalar::min);
            (0..polygon.len()).for_each(|i| {
                let (a, b) = (polygon[i], polygon[(i + 1) % polygon.len()]);
                if a != b
                    && !(rank(a) == 3 && rank(b) == 3)
                    && !(crossing_edge.contains_key(&a)
                        && crossing_edge.get(&a) == crossing_edge.get(&b))
                    && (&coordinates[b] - &coordinates[a]).norm() < limit
                {
                    let mut key = [a, b];
                    key.sort_unstable();
                    short.push(key);
                }
            })
        });
        short.sort_unstable();
        short.dedup();
        let mut parents = HashMap::new();
        let mut anchored = HashMap::new();
        fn root(parents: &mut HashMap<usize, usize>, node: usize) -> usize {
            let parent = *parents.entry(node).or_insert(node);
            if parent == node {
                node
            } else {
                let root = root(parents, parent);
                parents.insert(node, root);
                root
            }
        }
        short.into_iter().for_each(|[a, b]| {
            let (ra, rb) = (root(&mut parents, a), root(&mut parents, b));
            if ra != rb {
                let (ha, hb) = (
                    *anchored.entry(ra).or_insert(rank(ra) == 3),
                    *anchored.entry(rb).or_insert(rank(rb) == 3),
                );
                if !(ha && hb) {
                    let (keep, gone) = if ra < rb { (ra, rb) } else { (rb, ra) };
                    parents.insert(gone, keep);
                    anchored.insert(keep, ha || hb);
                }
            }
        });
        let mut clusters = HashMap::<usize, Vec<usize>>::new();
        let mut merging: Vec<usize> = parents.keys().copied().collect();
        merging.sort_unstable();
        merging.into_iter().for_each(|node| {
            let root = root(&mut parents, node);
            clusters.entry(root).or_default().push(node);
        });
        let mut ordered: Vec<Vec<usize>> = clusters
            .into_values()
            .filter(|cluster| cluster.len() > 1)
            .collect();
        ordered.sort_unstable();
        ordered.into_iter().for_each(|cluster| {
            let survivor = cluster
                .iter()
                .copied()
                .reduce(|best, node| if rank(node) > rank(best) { node } else { best })
                .unwrap();
            let affected: Vec<usize> = {
                let mut faces: Vec<usize> = cluster
                    .iter()
                    .filter_map(|node| node_faces.get(node))
                    .flatten()
                    .copied()
                    .collect();
                faces.sort_unstable();
                faces.dedup();
                faces
            };
            if affected.is_empty() {
                return;
            }
            let cells: Vec<usize> = {
                let mut cells: Vec<usize> = affected
                    .iter()
                    .flat_map(|face| face_cells[face].iter().copied())
                    .filter(|&cell| alive[cell])
                    .collect();
                cells.sort_unstable();
                cells.dedup();
                cells
            };
            let oriented = |cell: usize, updated: &HashMap<usize, Vec<usize>>| -> Vec<Vec<usize>> {
                sets[cell]
                    .iter()
                    .filter_map(|face| {
                        let polygon = updated.get(face).unwrap_or(&faces_nodes[*face]);
                        (polygon.len() > 2).then(|| {
                            if owners[*face] == cell {
                                polygon.clone()
                            } else {
                                polygon.iter().rev().copied().collect()
                            }
                        })
                    })
                    .collect()
            };
            let volumes: Vec<Scalar> = cells
                .iter()
                .map(|&cell| signed_volume(&oriented(cell, &HashMap::new()), coordinates))
                .collect();
            let position = if rank(survivor) >= 2 {
                coordinates[survivor].clone()
            } else {
                let centroid = cluster
                    .iter()
                    .map(|&node| coordinates[node].clone())
                    .sum::<Coordinate<D>>()
                    / cluster.len() as Scalar;
                bvh.closest_point(&centroid, surface_coordinates, &surface_elements)
                    .map(|(point, _)| point)
                    .unwrap_or(centroid)
            };
            let previous = coordinates[survivor].clone();
            coordinates[survivor] = position;
            let mut pinched = false;
            let updated: HashMap<usize, Vec<usize>> = affected
                .iter()
                .map(|&face| {
                    let mut polygon = Vec::new();
                    faces_nodes[face]
                        .iter()
                        .map(|&node| {
                            if cluster.binary_search(&node).is_ok() {
                                survivor
                            } else {
                                node
                            }
                        })
                        .for_each(|node| {
                            if polygon.last() != Some(&node) {
                                polygon.push(node)
                            }
                        });
                    while polygon.len() > 1 && polygon.first() == polygon.last() {
                        polygon.pop();
                    }
                    let mut check = polygon.clone();
                    check.sort_unstable();
                    check.dedup();
                    if check.len() != polygon.len() {
                        pinched = true;
                    }
                    (face, polygon)
                })
                .collect();
            let valid = !pinched
                && cells.iter().zip(volumes).all(|(&cell, volume)| {
                    let bound = COLLAPSE_FRACTION * COLLAPSE_FRACTION * scales[cell].powi(3);
                    let faces = oriented(cell, &updated);
                    if faces.is_empty() {
                        return volume <= bound;
                    }
                    let mut keys: Vec<Vec<usize>> = faces
                        .iter()
                        .map(|face| {
                            let mut key = face.clone();
                            key.sort_unstable();
                            key
                        })
                        .collect();
                    keys.sort_unstable();
                    let count = keys.len();
                    keys.dedup();
                    let new = signed_volume(&faces, coordinates);
                    count == keys.len() && count > 3 && new > 0.0 && (new - volume).abs() <= bound
                });
            if !valid {
                coordinates[survivor] = previous;
                return;
            }
            updated.into_iter().for_each(|(face, polygon)| {
                if polygon.len() > 2 {
                    faces_nodes[face] = polygon;
                    node_faces.entry(survivor).or_default().insert(face);
                } else {
                    faces_nodes[face].iter().for_each(|node| {
                        if let Some(incident) = node_faces.get_mut(node) {
                            incident.remove(&face);
                        }
                    });
                    face_cells
                        .remove(&face)
                        .into_iter()
                        .flatten()
                        .for_each(|cell| {
                            sets[cell].remove(&face);
                            if sets[cell].is_empty() {
                                alive[cell] = false;
                            }
                        });
                }
            });
            cluster.into_iter().for_each(|node| {
                if node != survivor {
                    node_faces.remove(&node);
                }
            });
        });
    }
}
