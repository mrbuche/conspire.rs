#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::BoundingBox,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, hash_map::Entry},
};

const CROSSING_TOLERANCE: Scalar = 1.0e-8;
const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    Inside,
    Cut,
    Outside,
}

pub struct Tables {
    signs: HashMap<usize, bool>,
    crossings: HashMap<[usize; 2], Coordinate<D>>,
    faces: HashMap<Vec<usize>, [usize; 4]>,
    segments: HashMap<Vec<usize>, Vec<[[usize; 2]; 2]>>,
}

impl Tables {
    pub fn signs(&self) -> &HashMap<usize, bool> {
        &self.signs
    }
    pub fn crossings(&self) -> &HashMap<[usize; 2], Coordinate<D>> {
        &self.crossings
    }
    pub fn faces(&self) -> &HashMap<Vec<usize>, [usize; 4]> {
        &self.faces
    }
    pub fn segments(&self) -> &HashMap<Vec<usize>, Vec<[[usize; 2]; 2]>> {
        &self.segments
    }
}

fn clip_face(
    corners: &[usize; 4],
    signs: &HashMap<usize, bool>,
    crossing_ids: &HashMap<[usize; 2], usize>,
    segments: Option<&Vec<[[usize; 2]; 2]>>,
) -> Vec<Vec<usize>> {
    let edge_keys: [[usize; 2]; 4] = from_fn(|i| {
        let mut key = [corners[i], corners[(i + 1) % 4]];
        key.sort_unstable();
        key
    });
    let crossed = edge_keys
        .iter()
        .filter(|key| crossing_ids.contains_key(*key))
        .count();
    let walk = || {
        vec![
            (0..4)
                .flat_map(|i| {
                    let mut items = Vec::new();
                    if signs[&corners[i]] {
                        items.push(corners[i])
                    }
                    if let Some(&crossing) = crossing_ids.get(&edge_keys[i]) {
                        items.push(crossing)
                    }
                    items
                })
                .collect(),
        ]
    };
    match crossed {
        0 => {
            if signs[&corners[0]] {
                vec![corners.to_vec()]
            } else {
                vec![]
            }
        }
        2 => walk(),
        4 => {
            let pairs = segments.unwrap();
            let shared = |pair: &[[usize; 2]; 2]| {
                *pair[0].iter().find(|node| pair[1].contains(node)).unwrap()
            };
            if signs[&shared(&pairs[0])] {
                pairs
                    .iter()
                    .map(|pair| {
                        let corner = shared(pair);
                        let at = corners.iter().position(|&node| node == corner).unwrap();
                        vec![
                            crossing_ids[&edge_keys[(at + 3) % 4]],
                            corner,
                            crossing_ids[&edge_keys[at]],
                        ]
                    })
                    .collect()
            } else {
                walk()
            }
        }
        _ => unreachable!(),
    }
}

fn contained(mesh: &Mesh<D>, classes: &[Class]) -> bool {
    let mut faces: HashMap<Vec<usize>, (usize, u8)> = HashMap::new();
    let mut offset = 0;
    mesh.iter().for_each(|block| {
        let local_faces = block.local_faces();
        block.iter().enumerate().for_each(|(local, element)| {
            local_faces.iter().for_each(|face| {
                let mut key: Vec<usize> = face.iter().map(|&local| element[local]).collect();
                key.sort_unstable();
                faces
                    .entry(key)
                    .and_modify(|(_, count)| *count += 1)
                    .or_insert((offset + local, 1));
            })
        });
        offset += block.number_of_elements();
    });
    faces
        .into_values()
        .all(|(owner, count)| count != 1 || classes[owner] == Class::Outside)
}

impl Tessellation {
    pub fn cut(&self, balancing: Balancing, scale: Scalar) -> Result<Mesh<D>, &'static str> {
        let mut octree =
            Octree::<u16, usize>::from_features(self, scale, CurvatureSizing::default(), PADDING);
        octree.equilibrate(balancing, Pairing::Regular)?;
        let mesh = octree.dualize();
        let classes = self.classify(&mesh);
        if !contained(&mesh, &classes) {
            return Err("tessellation is not contained within the dual mesh");
        }
        let tables = self.tables(&mesh, &classes)?;
        self.assemble(&mesh, &classes, &tables)
    }
    pub fn classify(&self, mesh: &Mesh<D>) -> Vec<Class> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let number_of_elements = mesh.number_of_elements();
        let mut cut = vec![false; number_of_elements];
        mesh.iter()
            .flat_map(|block| block.iter())
            .zip(cut.iter_mut())
            .for_each(|(element, flag)| {
                let bbox: BoundingBox<D> = element
                    .iter()
                    .map(|&node| &coordinates[node])
                    .collect::<CoordinatesRef<'_, D>>()
                    .into();
                *flag = bvh.overlapping(&bbox).into_iter().any(|triangle| {
                    let nodes = elements[triangle];
                    bbox.overlaps_triangle(
                        &surface_coordinates[nodes[0]],
                        &surface_coordinates[nodes[1]],
                        &surface_coordinates[nodes[2]],
                    )
                })
            });
        let mut faces = HashMap::new();
        let mut neighbors: Vec<Vec<usize>> = vec![Vec::new(); number_of_elements];
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().enumerate().for_each(|(local, element)| {
                let index = offset + local;
                if !cut[index] {
                    local_faces.iter().for_each(|face| {
                        let mut key: Vec<usize> =
                            face.iter().map(|&local| element[local]).collect();
                        key.sort_unstable();
                        match faces.entry(key) {
                            Entry::Occupied(entry) => {
                                let other = *entry.get();
                                neighbors[index].push(other);
                                neighbors[other].push(index);
                            }
                            Entry::Vacant(slot) => {
                                slot.insert(index);
                            }
                        }
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let centroids = mesh.centroids();
        let mut classes: Vec<Class> = cut
            .iter()
            .map(|&flag| if flag { Class::Cut } else { Class::Outside })
            .collect();
        let mut visited = cut;
        let mut stack = Vec::new();
        (0..number_of_elements).for_each(|seed| {
            if !visited[seed] {
                let class = if self.encloses(
                    &centroids[seed],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                ) {
                    Class::Inside
                } else {
                    Class::Outside
                };
                visited[seed] = true;
                stack.push(seed);
                while let Some(index) = stack.pop() {
                    classes[index] = class;
                    neighbors[index].iter().for_each(|&next| {
                        if !visited[next] {
                            visited[next] = true;
                            stack.push(next);
                        }
                    })
                }
            }
        });
        classes
    }
    pub fn tables(&self, mesh: &Mesh<D>, classes: &[Class]) -> Result<Tables, &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut edges = HashSet::new();
        let mut face_loops = HashMap::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            let local_faces = block.local_faces();
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    EDGES.iter().for_each(|&[a, b]| {
                        let mut key = [element[a], element[b]];
                        key.sort_unstable();
                        edges.insert(key);
                    });
                    local_faces.iter().for_each(|face| {
                        let corners: [usize; 4] = from_fn(|i| element[face[i]]);
                        let mut key = corners.to_vec();
                        key.sort_unstable();
                        face_loops.entry(key).or_insert(corners);
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let mut signs = HashMap::new();
        edges.iter().flatten().for_each(|&node| {
            signs.entry(node).or_insert_with(|| {
                self.encloses(
                    &coordinates[node],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                )
            });
        });
        let mut crossings = HashMap::new();
        edges.iter().try_for_each(|&[a, b]| {
            let span = &coordinates[b] - &coordinates[a];
            let length = span.norm();
            let hit = bvh.intersect(
                &(coordinates[a].clone(), span.clone()).into(),
                surface_coordinates,
                &elements,
            );
            if signs[&a] != signs[&b] {
                let near = hit.ok_or("crossing missing on a sign-change edge")?;
                let far = bvh
                    .intersect(
                        &(coordinates[b].clone(), &coordinates[a] - &coordinates[b]).into(),
                        surface_coordinates,
                        &elements,
                    )
                    .ok_or("crossing missing on a sign-change edge")?;
                if (length - near.distance() - far.distance()).abs()
                    > CROSSING_TOLERANCE.max(GRAZING_TOLERANCE * length)
                {
                    return Err("edge crosses the tessellation more than once");
                }
                crossings.insert(
                    [a, b],
                    &coordinates[a] + &(&span * (near.distance() / length)),
                );
            } else if hit.is_some_and(|near| near.distance() <= length) {
                return Err("edge crosses the tessellation more than once");
            }
            Ok(())
        })?;
        let mut segments = HashMap::new();
        face_loops.iter().try_for_each(|(key, &[a, b, c, d])| {
            let loop_edges: [[usize; 2]; 4] = [[a, b], [b, c], [c, d], [d, a]].map(|mut edge| {
                edge.sort_unstable();
                edge
            });
            let crossed: Vec<usize> = (0..4)
                .filter(|&i| crossings.contains_key(&loop_edges[i]))
                .collect();
            match crossed[..] {
                [] => Ok(()),
                [first, second] => {
                    segments.insert(key.clone(), vec![[loop_edges[first], loop_edges[second]]]);
                    Ok(())
                }
                [..] if crossed.len() == 4 => {
                    let center = [a, b, c, d]
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .sum::<Coordinate<D>>()
                        / 4.0;
                    let pairs = if self.encloses(
                        &center,
                        surface_coordinates,
                        &elements,
                        &normals,
                        &directions,
                    ) == signs[&a]
                    {
                        vec![
                            [loop_edges[0], loop_edges[1]],
                            [loop_edges[2], loop_edges[3]],
                        ]
                    } else {
                        vec![
                            [loop_edges[3], loop_edges[0]],
                            [loop_edges[1], loop_edges[2]],
                        ]
                    };
                    segments.insert(key.clone(), pairs);
                    Ok(())
                }
                _ => Err("inconsistent crossings around a face"),
            }
        })?;
        Ok(Tables {
            signs,
            crossings,
            faces: face_loops,
            segments,
        })
    }
    fn assemble(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        tables: &Tables,
    ) -> Result<Mesh<D>, &'static str> {
        let mut coordinates = mesh.coordinates().clone();
        let mut crossing_ids = HashMap::new();
        let mut crossed_edges: Vec<&[usize; 2]> = tables.crossings.keys().collect();
        crossed_edges.sort_unstable();
        crossed_edges.into_iter().for_each(|edge| {
            crossing_ids.insert(*edge, coordinates.len());
            coordinates.push(tables.crossings[edge].clone());
        });
        let face_polygons: HashMap<&Vec<usize>, Vec<Vec<usize>>> = tables
            .faces
            .iter()
            .map(|(key, corners)| {
                (
                    key,
                    clip_face(
                        corners,
                        &tables.signs,
                        &crossing_ids,
                        tables.segments.get(key),
                    ),
                )
            })
            .collect();
        let mut face_ids: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut faces_nodes: Vec<Vec<usize>> = Vec::new();
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
        let mut hexes: Vec<[usize; 8]> = Vec::new();
        let mut elements_faces: Vec<Vec<usize>> = Vec::new();
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
                        let cell_edges: Vec<[usize; 2]> = EDGES
                            .iter()
                            .map(|&[a, b]| {
                                let mut key = [element[a], element[b]];
                                key.sort_unstable();
                                key
                            })
                            .filter(|key| crossing_ids.contains_key(key))
                            .collect();
                        if cell_edges.is_empty() {
                            if tables.signs[&element[0]] {
                                hexes.push(from_fn(|i| element[i]));
                            }
                            return Ok(());
                        }
                        let mut cell_faces = Vec::new();
                        let mut adjacency: HashMap<[usize; 2], Vec<[usize; 2]>> = HashMap::new();
                        local_faces.iter().for_each(|face| {
                            let mut key: Vec<usize> =
                                face.iter().map(|&local| element[local]).collect();
                            key.sort_unstable();
                            face_polygons[&key].iter().for_each(|polygon| {
                                cell_faces.push(intern(
                                    &mut face_ids,
                                    &mut faces_nodes,
                                    polygon.clone(),
                                ))
                            });
                            if let Some(pairs) = tables.segments.get(&key) {
                                pairs.iter().for_each(|&[one, two]| {
                                    adjacency.entry(one).or_default().push(two);
                                    adjacency.entry(two).or_default().push(one);
                                })
                            }
                        });
                        if adjacency.values().any(|partners| partners.len() != 2) {
                            return Err("open cut chain within a cell");
                        }
                        let mut visited: HashSet<[usize; 2]> = HashSet::new();
                        cell_edges.iter().for_each(|&start| {
                            if visited.insert(start) {
                                let mut polygon = vec![crossing_ids[&start]];
                                let mut previous = start;
                                let mut current = adjacency[&start][0];
                                while current != start {
                                    visited.insert(current);
                                    polygon.push(crossing_ids[&current]);
                                    let next = if adjacency[&current][0] == previous {
                                        adjacency[&current][1]
                                    } else {
                                        adjacency[&current][0]
                                    };
                                    previous = current;
                                    current = next;
                                }
                                cell_faces.push(intern(&mut face_ids, &mut faces_nodes, polygon));
                            }
                        });
                        let mut roots: HashMap<usize, usize> = HashMap::new();
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
                        cell_faces.iter().for_each(|&face| {
                            let root = find(&mut roots, faces_nodes[face][0]);
                            faces_nodes[face][1..].iter().for_each(|&node| {
                                let other = find(&mut roots, node);
                                roots.insert(other, root);
                            })
                        });
                        let nodes: HashSet<usize> = cell_faces
                            .iter()
                            .flat_map(|&face| faces_nodes[face].iter().copied())
                            .collect();
                        let components: HashSet<usize> = nodes
                            .into_iter()
                            .map(|node| find(&mut roots, node))
                            .collect();
                        if components.len() != 1 {
                            return Err("disconnected cell interior requires refinement");
                        }
                        elements_faces.push(cell_faces);
                        Ok(())
                    }
                }
            })?;
            offset += block.number_of_elements();
            Ok(())
        })?;
        let mut remap: HashMap<usize, usize> = HashMap::new();
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
    fn encloses(
        &self,
        point: &Coordinate<D>,
        surface_coordinates: &Coordinates<D>,
        elements: &[&[usize]],
        normals: &[&Coordinate<D>],
        directions: &[Coordinate<D>; 3],
    ) -> bool {
        directions
            .iter()
            .find_map(|direction| {
                let ray = (point.clone(), direction.clone()).into();
                match self.bvh().intersect(&ray, surface_coordinates, elements) {
                    None => Some(false),
                    Some(hit) => {
                        let normal = normals[hit.index()];
                        let cosine = (direction * normal) / normal.norm();
                        (cosine.abs() > GRAZING_TOLERANCE).then_some(cosine > 0.0)
                    }
                }
            })
            .unwrap_or(false)
    }
}
