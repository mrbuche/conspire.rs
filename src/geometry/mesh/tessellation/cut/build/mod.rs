#[cfg(test)]
mod test;

use super::{
    Class, SLIVER_FRACTION,
    face::{clip_face, face_cut},
    geometry::{face_area, signed_volume},
    split::{Split, split_cell},
    tables::GenericTables,
    topology::{element_edges, face_owners, oriented_element_faces},
};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh, tessellation::D},
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

/// The generic (arbitrary-polyhedra) analogue of `assemble()`: runs
/// `split_cell` over every cell of the mesh and interns the results into a
/// single polyhedral mesh. Unlike `assemble()`, it performs no sliver
/// agglomeration, no short-edge collapse, and no hex-recomposition —
/// those remain hex-only refinements layered on top of the working path
/// once this generic core is proven on real polyhedra.
pub(super) fn assemble_generic(
    mesh: &Mesh<D>,
    classes: &[Class],
    tables: &GenericTables,
) -> Result<Mesh<D>, &'static str> {
    let mut coordinates = mesh.coordinates().clone();
    let mut crossing_ids = HashMap::<[usize; 2], Vec<usize>>::new();
    let mut crossed_edges: Vec<&[usize; 2]> = tables.crossings().keys().collect();
    crossed_edges.sort_unstable();
    crossed_edges.into_iter().for_each(|edge| {
        let ids: Vec<usize> = tables.crossings()[edge]
            .iter()
            .map(|point| {
                coordinates.push(point.clone());
                coordinates.len() - 1
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
    let mut offset = 0;
    mesh.iter().try_for_each(|block| {
        let owners = face_owners(block);
        block
            .iter()
            .enumerate()
            .try_for_each(|(local, element)| -> Result<(), &'static str> {
                let cell = elements_faces.len();
                let mut emit = |polygons: Vec<Vec<usize>>, fraction: Scalar| {
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
                };
                match classes[offset + local] {
                    Class::Outside => {}
                    Class::Inside => emit(
                        oriented_element_faces(block, element, local, owners.as_deref()),
                        1.0,
                    ),
                    Class::Cut => {
                        let faces =
                            oriented_element_faces(block, element, local, owners.as_deref());
                        let edges = element_edges(&faces);
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
                            Split::Unchanged => emit(faces, 1.0),
                            Split::Cut(cut_cell) => {
                                let mut polygons = cut_cell.polygons;
                                orient_outward(&mut polygons, cut_cell.clipped, &coordinates);
                                let fraction = signed_volume(&polygons, &coordinates)
                                    / signed_volume(&faces, &coordinates);
                                emit(polygons, fraction)
                            }
                        }
                    }
                }
                Ok(())
            })?;
        offset += block.number_of_elements();
        Ok(())
    })?;
    let mut sets: Vec<HashSet<usize>> = elements_faces
        .iter()
        .map(|faces| faces.iter().copied().collect())
        .collect();
    let alive = agglomerate(
        &mut sets,
        &mut face_owner,
        &faces_nodes,
        &fractions,
        &coordinates,
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
