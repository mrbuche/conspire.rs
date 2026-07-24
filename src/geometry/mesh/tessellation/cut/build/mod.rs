#[cfg(test)]
mod test;

use super::{
    COLLAPSE_FRACTION, Class, SLIVER_FRACTION, Sign,
    face::{clip_face, face_cut},
    geometry::{face_area, signed_volume},
    split::{Split, split_cell},
    tables::GenericTables,
    topology::{element_edges, face_owners, oriented_element_faces},
};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
    },
    math::{CrossProduct, Scalar, Tensor, TensorVec},
};
use std::collections::{HashMap, HashSet};

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

/// Absorbs each cut cell retaining less than `SLIVER_FRACTION` of its source
/// cell's volume into whichever live neighbour it shares the most face area
/// with, worst sliver first; the generic analogue of `assemble()`'s
/// equivalent pass. Faces shared by the pair become interior and vanish.
fn agglomerate(
    sets: &mut [HashSet<usize>],
    owners: &mut [usize],
    faces_nodes: &[Vec<usize>],
    fractions: &[Scalar],
    coordinates: &Coordinates<D>,
) -> Vec<bool> {
    let mut face_cells = HashMap::<usize, Vec<usize>>::new();
    sets.iter().enumerate().for_each(|(cell, faces)| {
        faces
            .iter()
            .for_each(|&face| face_cells.entry(face).or_default().push(cell))
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
            face_cells[&face].iter().for_each(|&other| {
                if other != sliver && alive[other] {
                    *areas.entry(other).or_insert(0.0) +=
                        face_area(&faces_nodes[face], coordinates);
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
                face_cells
                    .get_mut(&face)
                    .unwrap()
                    .iter_mut()
                    .for_each(|cell| {
                        if *cell == sliver {
                            *cell = target
                        }
                    })
            });
            alive[sliver] = false
        }
    });
    alive
}

/// Orients newly created interior cut faces (`polygons[clipped..]`) outward
/// from the whole cell's centroid; mirrors the hex-only `assemble()`'s
/// equivalent step, generalized to arbitrary polygon vertex counts.
fn orient_outward(polygons: &mut [Vec<usize>], clipped: usize, coordinates: &Coordinates<D>) {
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
                let two = &coordinates[polygon[(i + 1) % polygon.len()]] - &centroid;
                one.cross(&two) * &middle
            })
            .sum();
        if outward < 0.0 {
            polygon.reverse()
        }
    });
}

impl Tessellation {
    /// The generic (arbitrary-polyhedra) analogue of `assemble()`: runs
    /// `split_cell` over every cell of the mesh, interns the results into a
    /// single polyhedral mesh, then agglomerates slivers and collapses short
    /// edges as `assemble()` does. Hex recomposition is deliberately omitted,
    /// having no purpose for polyhedral output.
    pub(super) fn assemble_generic(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        tables: &GenericTables,
    ) -> Result<Mesh<D>, &'static str> {
        let mut coordinates = mesh.coordinates().clone();
        let mut crossing_ids = HashMap::<[usize; 2], Vec<usize>>::new();
        let mut crossing_edge = HashMap::<usize, [usize; 2]>::new();
        let mut crossed_edges: Vec<&[usize; 2]> = tables.crossings().keys().collect();
        crossed_edges.sort_unstable();
        crossed_edges.into_iter().for_each(|edge| {
            let ids: Vec<usize> = tables.crossings()[edge]
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
        let mut face_cuts = HashMap::new();
        let mut face_polygons = HashMap::new();
        tables
            .faces()
            .iter()
            .try_for_each(|(key, corners)| -> Result<(), &'static str> {
                let cut = face_cut(corners, tables.signs(), tables.crossings())?;
                let polygons = if cut.flush {
                    Vec::new()
                } else {
                    clip_face(&cut, tables.segments().get(key), &crossing_ids)
                };
                face_polygons.insert(key.clone(), polygons);
                face_cuts.insert(key.clone(), cut);
                Ok(())
            })?;
        let mut face_ids = HashMap::<Vec<usize>, usize>::new();
        let mut faces_nodes = Vec::new();
        let mut face_owner = Vec::new();
        let mut elements_faces = Vec::<Vec<usize>>::new();
        let mut fractions = Vec::<Scalar>::new();
        let mut scales = Vec::<Scalar>::new();
        let mut whole = HashSet::<usize>::new();
        let mut offset = 0;
        mesh.iter().try_for_each(|block| {
            let owners = face_owners(block);
            block.iter().enumerate().try_for_each(
                |(local, element)| -> Result<(), &'static str> {
                    let cell = elements_faces.len();
                    let mut emit = |polygons: Vec<Vec<usize>>, fraction: Scalar, scale: Scalar| {
                        let ids: Vec<usize> = polygons
                            .into_iter()
                            .map(|polygon| {
                                intern(
                                    &mut face_ids,
                                    &mut faces_nodes,
                                    &mut face_owner,
                                    polygon,
                                    cell,
                                )
                            })
                            .collect();
                        elements_faces.push(ids);
                        fractions.push(fraction);
                        scales.push(scale);
                    };
                    let shortest = |edges: &[[usize; 2]], coordinates: &Coordinates<D>| {
                        edges
                            .iter()
                            .map(|&[a, b]| (&coordinates[b] - &coordinates[a]).norm())
                            .fold(Scalar::INFINITY, Scalar::min)
                    };
                    match classes[offset + local] {
                        Class::Outside => {}
                        Class::Inside => {
                            let faces =
                                oriented_element_faces(block, element, local, owners.as_deref());
                            let scale = shortest(&element_edges(&faces), &coordinates);
                            faces.iter().flatten().for_each(|&node| {
                                whole.insert(node);
                            });
                            emit(faces, 1.0, scale)
                        }
                        Class::Cut => {
                            let faces =
                                oriented_element_faces(block, element, local, owners.as_deref());
                            let edges = element_edges(&faces);
                            let scale = shortest(&edges, &coordinates);
                            match split_cell(
                                &faces,
                                &edges,
                                tables.signs(),
                                &face_cuts,
                                tables.faces(),
                                &face_polygons,
                                tables.segments(),
                                &crossing_ids,
                            )? {
                                Split::Discarded => {}
                                Split::Unchanged => {
                                    faces.iter().flatten().for_each(|&node| {
                                        whole.insert(node);
                                    });
                                    emit(faces, 1.0, scale)
                                }
                                Split::Cut(cut_cell) => {
                                    let mut polygons = cut_cell.polygons;
                                    orient_outward(&mut polygons, cut_cell.clipped, &coordinates);
                                    let fraction = signed_volume(&polygons, &coordinates)
                                        / signed_volume(&faces, &coordinates);
                                    emit(polygons, fraction, scale)
                                }
                            }
                        }
                    }
                    Ok(())
                },
            )?;
            offset += block.number_of_elements();
            Ok(())
        })?;
        let mut sets: Vec<HashSet<usize>> = elements_faces
            .iter()
            .map(|faces| faces.iter().copied().collect())
            .collect();
        let mut alive = agglomerate(
            &mut sets,
            &mut face_owner,
            &faces_nodes,
            &fractions,
            &coordinates,
        );
        self.collapse_short_edges_generic(
            &mut coordinates,
            &mut faces_nodes,
            &mut sets,
            &face_owner,
            &mut alive,
            tables,
            &whole,
            &scales,
            &crossing_edge,
        );
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
                                if face_owner[face] != cell {
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
        let mut remap = HashMap::new();
        let mut points = Coordinates::new();
        let mut renumber = |node: usize, points: &mut Coordinates<D>| -> usize {
            *remap.entry(node).or_insert_with(|| {
                points.push(coordinates[node].clone());
                points.len() - 1
            })
        };
        let faces_nodes: Vec<Vec<usize>> = compact_nodes
            .into_iter()
            .map(|face| {
                face.into_iter()
                    .map(|node| renumber(node, &mut points))
                    .collect()
            })
            .collect();
        Ok((
            vec![Connectivity::Polyhedral(
                (elements_faces, faces_nodes).into(),
            )],
            points,
        )
            .into())
    }

    /// The generic analogue of `collapse_short_edges()`: merges clusters of
    /// near-coincident nodes, taking the length yardstick from each cell's
    /// own source cell (`scales`) and treating nodes of uncut cells
    /// (`whole`) as anchored, in place of the hex-only source/hex arrays.
    #[allow(clippy::too_many_arguments)]
    fn collapse_short_edges_generic(
        &self,
        coordinates: &mut Coordinates<D>,
        faces_nodes: &mut [Vec<usize>],
        sets: &mut [HashSet<usize>],
        owners: &[usize],
        alive: &mut [bool],
        tables: &GenericTables,
        whole: &HashSet<usize>,
        scales: &[Scalar],
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
        let mut ranks = HashMap::new();
        tables.signs().iter().for_each(|(&node, &sign)| {
            ranks.insert(node, if sign == Sign::On { 2 } else { 1 });
        });
        whole.iter().for_each(|&node| {
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
