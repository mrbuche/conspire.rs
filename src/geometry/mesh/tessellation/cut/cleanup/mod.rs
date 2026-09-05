use super::{
    COLLAPSE_FRACTION, SLIVER_FRACTION, Sign,
    geometry::{face_area, signed_volume},
};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::tessellation::{D, Tessellation},
    },
    math::{Quantity, Scalar, Tensor},
    units::{Length, Volume},
};
use std::collections::{HashMap, HashSet};

pub(super) fn intern(
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

pub(super) fn agglomerate(
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
                    *areas.entry(other).or_insert(Quantity::default()) +=
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

impl Tessellation {
    #[expect(clippy::too_many_arguments)]
    pub(super) fn collapse_short_edges(
        &self,
        coordinates: &mut Coordinates<D>,
        faces_nodes: &mut [Vec<usize>],
        sets: &mut [HashSet<usize>],
        owners: &[usize],
        alive: &mut [bool],
        signs: &HashMap<usize, Sign>,
        whole: &HashSet<usize>,
        scales: &[Quantity<Length>],
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
        signs.iter().for_each(|(&node, &sign)| {
            ranks.insert(node, if sign == Sign::On { 2 } else { 1 });
        });
        whole.iter().for_each(|&node| {
            ranks.insert(node, 3);
        });
        let rank = |node: usize| ranks.get(&node).copied().unwrap_or(0);
        let mut short = Vec::new();
        face_cells.iter().for_each(|(&face, cells)| {
            let polygon = &faces_nodes[face];
            let limit = cells
                .iter()
                .map(|&cell| scales[cell])
                .fold(Quantity::new(Scalar::INFINITY), Quantity::min)
                * COLLAPSE_FRACTION;
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
            let volumes: Vec<Quantity<Volume>> = cells
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
                    let scale = scales[cell];
                    let bound = scale * scale * scale * (COLLAPSE_FRACTION * COLLAPSE_FRACTION);
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
                    count == keys.len()
                        && count > 3
                        && new > Quantity::default()
                        && (new - volume).abs() <= bound
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
