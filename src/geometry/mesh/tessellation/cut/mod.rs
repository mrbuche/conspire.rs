#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates, CoordinatesRef,
        bbox::BoundingBox,
        mesh::{
            Connectivity, Mesh,
            quality::metrics::{Kind, minimum_scaled_jacobian},
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, CurvatureSizing, Dualization, Octree, Pairing},
    },
    math::{CrossProduct, Scalar, Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet, hash_map::Entry},
    mem::take,
};

const CROSSING_TOLERANCE: Scalar = 1.0e-8;
const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const PADDING: u16 = 2;
const SLIVER_FRACTION: Scalar = 0.1;
const SNAP_HARD: Scalar = 0.05;
const SNAP_SOFT: Scalar = 0.2;
const FACES: [[usize; 4]; 6] = [
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
    [0, 3, 2, 1],
    [4, 5, 6, 7],
];
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sign {
    Inside,
    On,
    Outside,
}

pub struct Tables {
    signs: HashMap<usize, Sign>,
    crossings: HashMap<[usize; 2], Coordinate<D>>,
    faces: HashMap<Vec<usize>, [usize; 4]>,
    segments: HashMap<Vec<usize>, Vec<[[usize; 2]; 2]>>,
}

impl Tables {
    pub fn signs(&self) -> &HashMap<usize, Sign> {
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

struct FaceCut {
    endpoints: Vec<[usize; 2]>,
    sides: Vec<Sign>,
    interiors: Vec<Vec<usize>>,
    emitted: Vec<usize>,
    on_edges: Vec<([usize; 2], Sign)>,
    inside: bool,
    flush: bool,
}

fn face_cut(
    corners: &[usize; 4],
    signs: &HashMap<usize, Sign>,
    crossings: &HashMap<[usize; 2], Coordinate<D>>,
) -> Result<FaceCut, &'static str> {
    let statuses: [Sign; 4] = from_fn(|i| signs[&corners[i]]);
    let edge_keys: [[usize; 2]; 4] = from_fn(|i| {
        let mut key = [corners[i], corners[(i + 1) % 4]];
        key.sort_unstable();
        key
    });
    let crossed: [bool; 4] = from_fn(|i| crossings.contains_key(&edge_keys[i]));
    let flip = |sign| {
        if sign == Sign::Inside {
            Sign::Outside
        } else {
            Sign::Inside
        }
    };
    let decisive: Vec<usize> = (0..4).filter(|&i| statuses[i] != Sign::On).collect();
    let Some(&start) = decisive.first() else {
        return Ok(FaceCut {
            endpoints: Vec::new(),
            sides: Vec::new(),
            interiors: Vec::new(),
            emitted: corners.to_vec(),
            on_edges: Vec::new(),
            inside: false,
            flush: true,
        });
    };
    let mut pass = [false; 4];
    for (w, &from) in decisive.iter().enumerate() {
        let to = decisive[(w + 1) % decisive.len()];
        let mut ons = Vec::new();
        let mut at = (from + 1) % 4;
        while at != to {
            ons.push(at);
            at = (at + 1) % 4;
        }
        let change = statuses[from] != statuses[to];
        if ons.is_empty() {
            if crossed[from] != change {
                return Err("inconsistent signs around a face");
            }
        } else if change {
            pass[if statuses[from] == Sign::Inside {
                ons[0]
            } else {
                *ons.last().unwrap()
            }] = true;
        }
    }
    let mut side = statuses[start];
    let mut endpoints = Vec::new();
    let mut sides = Vec::new();
    let mut interiors = Vec::new();
    let mut current = Vec::new();
    let mut prefix = Vec::new();
    let mut opened = false;
    let endpoint = |key: [usize; 2],
                    side: &mut Sign,
                    current: &mut Vec<usize>,
                    endpoints: &mut Vec<[usize; 2]>,
                    sides: &mut Vec<Sign>,
                    interiors: &mut Vec<Vec<usize>>,
                    prefix: &mut Vec<usize>,
                    opened: &mut bool| {
        if *opened {
            interiors.push(take(current));
        } else {
            *prefix = take(current);
            *opened = true;
        }
        endpoints.push(key);
        *side = flip(*side);
        sides.push(*side);
    };
    let mut on_edges = Vec::new();
    for step in 0..4 {
        let at = (start + step) % 4;
        match statuses[at] {
            Sign::Inside | Sign::Outside => {
                if statuses[at] != side {
                    return Err("inconsistent signs around a face");
                }
                if side == Sign::Inside {
                    current.push(corners[at])
                }
            }
            Sign::On => {
                if pass[at] {
                    endpoint(
                        [corners[at], corners[at]],
                        &mut side,
                        &mut current,
                        &mut endpoints,
                        &mut sides,
                        &mut interiors,
                        &mut prefix,
                        &mut opened,
                    );
                } else if side == Sign::Inside {
                    current.push(corners[at])
                }
            }
        }
        if crossed[at] {
            endpoint(
                edge_keys[at],
                &mut side,
                &mut current,
                &mut endpoints,
                &mut sides,
                &mut interiors,
                &mut prefix,
                &mut opened,
            );
        } else if statuses[at] == Sign::On && statuses[(at + 1) % 4] == Sign::On {
            on_edges.push((edge_keys[at], side));
        }
    }
    let emitted = if opened {
        current.extend(prefix);
        interiors.push(current);
        Vec::new()
    } else {
        current
    };
    Ok(FaceCut {
        endpoints,
        sides,
        interiors,
        emitted,
        on_edges,
        inside: statuses.contains(&Sign::Inside),
        flush: false,
    })
}

fn clip_face(
    cut: &FaceCut,
    chords: Option<&Vec<[[usize; 2]; 2]>>,
    crossing_ids: &HashMap<[usize; 2], usize>,
) -> Vec<Vec<usize>> {
    let point = |key: [usize; 2]| {
        if key[0] == key[1] {
            key[0]
        } else {
            crossing_ids[&key]
        }
    };
    if cut.endpoints.is_empty() {
        return if cut.inside && cut.emitted.len() > 2 {
            vec![cut.emitted.clone()]
        } else {
            vec![]
        };
    }
    let mut partner = HashMap::new();
    chords.unwrap().iter().for_each(|&[one, two]| {
        partner.insert(one, two);
        partner.insert(two, one);
    });
    let arcs: HashMap<[usize; 2], usize> = cut
        .endpoints
        .iter()
        .enumerate()
        .map(|(index, &key)| (key, index))
        .collect();
    let count = cut.endpoints.len();
    let mut visited = vec![false; count];
    let mut polygons = Vec::new();
    (0..count).for_each(|origin| {
        if cut.sides[origin] == Sign::Inside && !visited[origin] {
            let mut polygon = vec![point(cut.endpoints[origin])];
            let mut arc = origin;
            loop {
                visited[arc] = true;
                polygon.extend(cut.interiors[arc].iter().copied());
                let end = cut.endpoints[(arc + 1) % count];
                polygon.push(point(end));
                let jump = arcs[&partner[&end]];
                if jump == origin {
                    break;
                }
                polygon.push(point(cut.endpoints[jump]));
                arc = jump;
            }
            if polygon.len() > 2 {
                polygons.push(polygon)
            }
        }
    });
    polygons
}

fn face_area(face: &[usize], coordinates: &Coordinates<D>) -> Scalar {
    let middle = face
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / face.len() as Scalar;
    (0..face.len())
        .map(|i| {
            let one = &coordinates[face[i]] - &middle;
            let two = &coordinates[face[(i + 1) % face.len()]] - &middle;
            one.cross(&two).norm() / 2.0
        })
        .sum()
}

fn star_volume(faces: &[Vec<usize>], coordinates: &Coordinates<D>) -> Scalar {
    let nodes: HashSet<usize> = faces.iter().flatten().copied().collect();
    let centroid = nodes
        .iter()
        .map(|&node| coordinates[node].clone())
        .sum::<Coordinate<D>>()
        / nodes.len() as Scalar;
    faces
        .iter()
        .map(|face| {
            let middle = &(face
                .iter()
                .map(|&node| coordinates[node].clone())
                .sum::<Coordinate<D>>()
                / face.len() as Scalar)
                - &centroid;
            (0..face.len())
                .map(|i| {
                    let one = &coordinates[face[i]] - &centroid;
                    let two = &coordinates[face[(i + 1) % face.len()]] - &centroid;
                    (one.cross(&two) * &middle).abs() / 6.0
                })
                .sum::<Scalar>()
        })
        .sum()
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
        let (mesh, snapped) = self.snap(mesh, &classes)?;
        let tables = self.tables(&mesh, &classes, &snapped)?;
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
    fn snap(
        &self,
        mesh: Mesh<D>,
        classes: &[Class],
    ) -> Result<(Mesh<D>, HashSet<usize>), &'static str> {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let bvh = self.bvh();
        let coordinates = mesh.coordinates();
        let mut lengths: HashMap<usize, Scalar> = HashMap::new();
        let mut offset = 0;
        mesh.iter().for_each(|block| {
            block.iter().enumerate().for_each(|(local, element)| {
                if classes[offset + local] == Class::Cut {
                    EDGES.iter().for_each(|&[a, b]| {
                        let length = (&coordinates[element[b]] - &coordinates[element[a]]).norm();
                        [element[a], element[b]].into_iter().for_each(|node| {
                            lengths
                                .entry(node)
                                .and_modify(|shortest| *shortest = shortest.min(length))
                                .or_insert(length);
                        })
                    })
                }
            });
            offset += block.number_of_elements();
        });
        let cells: Vec<Vec<usize>> = mesh
            .iter()
            .flat_map(|block| block.iter().map(|element| element.to_vec()))
            .collect();
        let incidents = mesh.node_element_connectivity().to_vec();
        let mut working = mesh.coordinates().clone();
        let mut snapped = HashSet::new();
        let mut candidates: Vec<usize> = lengths.keys().copied().collect();
        candidates.sort_unstable();
        candidates.into_iter().for_each(|node| {
            if let Some((closest, _)) =
                bvh.closest_point(&working[node], surface_coordinates, &elements)
            {
                let distance = (&closest - &working[node]).norm();
                let shortest = lengths[&node];
                let accept = if distance < SNAP_HARD * shortest {
                    true
                } else if distance < SNAP_SOFT * shortest {
                    let retained: Vec<&Vec<usize>> = incidents[node]
                        .iter()
                        .filter(|&&cell| classes[cell] == Class::Inside)
                        .map(|&cell| &cells[cell])
                        .collect();
                    let quality = |coordinates: &Coordinates<D>| {
                        retained
                            .iter()
                            .map(|cell| {
                                minimum_scaled_jacobian(Kind::Hexahedron, cell, coordinates)
                            })
                            .fold(Scalar::INFINITY, Scalar::min)
                    };
                    let before = quality(&working);
                    let previous = working[node].clone();
                    working[node] = closest.clone();
                    let keep = retained.is_empty() || quality(&working) >= before;
                    if !keep {
                        working[node] = previous;
                    }
                    keep
                } else {
                    false
                };
                if accept {
                    working[node] = closest;
                    snapped.insert(node);
                }
            }
        });
        let (connectivities, _) = mesh.into();
        let hexes: Vec<[usize; 8]> = Vec::try_from(connectivities)?;
        Ok((
            (vec![Connectivity::Hexahedral(hexes.into())], working).into(),
            snapped,
        ))
    }
    pub fn tables(
        &self,
        mesh: &Mesh<D>,
        classes: &[Class],
        snapped: &HashSet<usize>,
    ) -> Result<Tables, &'static str> {
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
                if snapped.contains(&node) {
                    Sign::On
                } else if self.encloses(
                    &coordinates[node],
                    surface_coordinates,
                    &elements,
                    &normals,
                    &directions,
                ) {
                    Sign::Inside
                } else {
                    Sign::Outside
                }
            });
        });
        let mut crossings = HashMap::new();
        edges.iter().try_for_each(|&[a, b]| {
            let span = &coordinates[b] - &coordinates[a];
            let length = span.norm();
            let margin = CROSSING_TOLERANCE.max(GRAZING_TOLERANCE * length);
            match (signs[&a], signs[&b]) {
                (Sign::On, Sign::On) => Ok(()),
                (Sign::On, _) | (_, Sign::On) => {
                    let (from, along) = if signs[&a] == Sign::On {
                        (b, &coordinates[a] - &coordinates[b])
                    } else {
                        (a, span)
                    };
                    if let Some(hit) = bvh.intersect(
                        &(coordinates[from].clone(), along).into(),
                        surface_coordinates,
                        &elements,
                    ) && hit.distance() < length - margin
                    {
                        return Err("edge crosses the tessellation more than once");
                    }
                    Ok(())
                }
                (inside, outside) if inside != outside => {
                    let near = bvh
                        .intersect(
                            &(coordinates[a].clone(), span.clone()).into(),
                            surface_coordinates,
                            &elements,
                        )
                        .ok_or("crossing missing on a sign-change edge")?;
                    let far = bvh
                        .intersect(
                            &(coordinates[b].clone(), &coordinates[a] - &coordinates[b]).into(),
                            surface_coordinates,
                            &elements,
                        )
                        .ok_or("crossing missing on a sign-change edge")?;
                    if (length - near.distance() - far.distance()).abs() > margin {
                        return Err("edge crosses the tessellation more than once");
                    }
                    crossings.insert(
                        [a, b],
                        &coordinates[a] + &(&span * (near.distance() / length)),
                    );
                    Ok(())
                }
                _ => {
                    if let Some(hit) = bvh.intersect(
                        &(coordinates[a].clone(), span).into(),
                        surface_coordinates,
                        &elements,
                    ) && hit.distance() <= length
                    {
                        return Err("edge crosses the tessellation more than once");
                    }
                    Ok(())
                }
            }
        })?;
        let contains = |point: &Coordinate<D>| {
            self.encloses(point, surface_coordinates, &elements, &normals, &directions)
        };
        let mut segments = HashMap::new();
        face_loops.iter().try_for_each(|(key, corners)| {
            let cut = face_cut(corners, &signs, &crossings)?;
            match cut.endpoints.len() {
                0 => Ok(()),
                2 => {
                    segments.insert(key.clone(), vec![[cut.endpoints[0], cut.endpoints[1]]]);
                    Ok(())
                }
                4 => {
                    let center = corners
                        .iter()
                        .map(|&node| coordinates[node].clone())
                        .sum::<Coordinate<D>>()
                        / 4.0;
                    let target = if contains(&center) {
                        Sign::Outside
                    } else {
                        Sign::Inside
                    };
                    let pairs: Vec<[[usize; 2]; 2]> = (0..4)
                        .filter(|&arc| cut.sides[arc] == target)
                        .map(|arc| [cut.endpoints[arc], cut.endpoints[(arc + 1) % 4]])
                        .collect();
                    if pairs.len() != 2 {
                        return Err("inconsistent crossings around a face");
                    }
                    segments.insert(key.clone(), pairs);
                    Ok(())
                }
                _ => Err("refinement required at a face"),
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
        let mut face_ids: HashMap<Vec<usize>, usize> = HashMap::new();
        let mut faces_nodes: Vec<Vec<usize>> = Vec::new();
        let mut owners: Vec<usize> = Vec::new();
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
        let point = |key: [usize; 2]| {
            if key[0] == key[1] {
                key[0]
            } else {
                crossing_ids[&key]
            }
        };
        let mut hexes: Vec<[usize; 8]> = Vec::new();
        let mut elements_faces: Vec<Vec<usize>> = Vec::new();
        let mut sources: Vec<[usize; 8]> = Vec::new();
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
                        let faces: Vec<(Vec<usize>, Vec<usize>)> = local_faces
                            .iter()
                            .map(|face| {
                                let oriented: Vec<usize> =
                                    face.iter().map(|&local| element[local]).collect();
                                let mut key = oriented.clone();
                                key.sort_unstable();
                                (key, oriented)
                            })
                            .collect();
                        let mut adjacency: HashMap<[usize; 2], Vec<[usize; 2]>> = HashMap::new();
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
                                if on_sides.contains(&Sign::Inside)
                                    && on_sides.contains(&Sign::Outside)
                                {
                                    adjacency.entry([na, na]).or_default().push([nb, nb]);
                                    adjacency.entry([nb, nb]).or_default().push([na, na]);
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
                        let mut polygons: Vec<Vec<usize>> = Vec::new();
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
                        let mut keys: Vec<[usize; 2]> = adjacency.keys().copied().collect();
                        keys.sort_unstable();
                        let mut visited: HashSet<[usize; 2]> = HashSet::new();
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
                                    intern(
                                        &mut face_ids,
                                        &mut faces_nodes,
                                        &mut owners,
                                        polygon,
                                        cell,
                                    )
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
        let mut face_polys: HashMap<usize, Vec<usize>> = HashMap::new();
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
            let mut areas: HashMap<usize, Scalar> = HashMap::new();
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
        let mut compacted: HashMap<usize, usize> = HashMap::new();
        let mut compact_nodes: Vec<Vec<usize>> = Vec::new();
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
