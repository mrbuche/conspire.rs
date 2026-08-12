#[cfg(test)]
mod test;

use super::{
    Class,
    cleanup::{agglomerate, intern},
    face::{clip_face, face_cut},
    geometry::signed_volume,
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
                    clip_face(
                        &cut,
                        tables.segments().get(key),
                        None,
                        &crossing_ids,
                        &HashMap::new(),
                    )
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
        self.collapse_short_edges(
            &mut coordinates,
            &mut faces_nodes,
            &mut sets,
            &face_owner,
            &mut alive,
            tables.signs(),
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
}
