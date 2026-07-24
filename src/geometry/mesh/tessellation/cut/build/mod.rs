#[cfg(test)]
mod test;

use super::{
    Class,
    face::{clip_face, face_cut},
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
    polygon: Vec<usize>,
) -> usize {
    let mut key = polygon.clone();
    key.sort_unstable();
    *face_ids.entry(key).or_insert_with(|| {
        faces_nodes.push(polygon);
        faces_nodes.len() - 1
    })
}

/// Orients newly created interior cut faces (`polygons[clipped..]`) outward
/// from the whole cell's centroid; mirrors the hex-only `assemble()`'s
/// equivalent step, generalized to arbitrary polygon vertex counts.
fn orient_outward(polygons: &mut [Vec<usize>], clipped: usize, coordinates: &Coordinates<D>) {
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
    let mut elements_faces = Vec::<Vec<usize>>::new();
    let mut offset = 0;
    mesh.iter().try_for_each(|block| {
        let owners = face_owners(block);
        block
            .iter()
            .enumerate()
            .try_for_each(|(local, element)| -> Result<(), &'static str> {
                match classes[offset + local] {
                    Class::Outside => {}
                    Class::Inside => {
                        let ids: Vec<usize> =
                            oriented_element_faces(block, element, local, owners.as_deref())
                                .into_iter()
                                .map(|polygon| intern(&mut face_ids, &mut faces_nodes, polygon))
                                .collect();
                        elements_faces.push(ids);
                    }
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
                            Split::Unchanged => {
                                let ids: Vec<usize> = faces
                                    .into_iter()
                                    .map(|polygon| intern(&mut face_ids, &mut faces_nodes, polygon))
                                    .collect();
                                elements_faces.push(ids);
                            }
                            Split::Cut(cut_cell) => {
                                let mut polygons = cut_cell.polygons;
                                orient_outward(&mut polygons, cut_cell.clipped, &coordinates);
                                let ids: Vec<usize> = polygons
                                    .into_iter()
                                    .map(|polygon| intern(&mut face_ids, &mut faces_nodes, polygon))
                                    .collect();
                                elements_faces.push(ids);
                            }
                        }
                    }
                }
                Ok(())
            })?;
        offset += block.number_of_elements();
        Ok(())
    })?;
    let mut remap = HashMap::new();
    let mut points = Coordinates::new();
    let mut renumber = |node: usize, points: &mut Coordinates<D>| -> usize {
        *remap.entry(node).or_insert_with(|| {
            points.push(coordinates[node].clone());
            points.len() - 1
        })
    };
    let faces_nodes: Vec<Vec<usize>> = faces_nodes
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
