#[cfg(test)]
mod test;

use super::{Sign, Vertex, face::FaceCut};
use std::collections::{HashMap, HashSet};

pub(super) struct CutCell {
    pub(super) polygons: Vec<Vec<usize>>,
    pub(super) clipped: usize,
}

pub(super) enum Split {
    Unchanged,
    Discarded,
    Cut(CutCell),
}

#[allow(clippy::too_many_arguments)]
pub(super) fn split_cell(
    faces: &[Vec<usize>],
    edges: &[[usize; 2]],
    signs: &HashMap<usize, Sign>,
    face_cuts: &HashMap<Vec<usize>, FaceCut>,
    face_orientations: &HashMap<Vec<usize>, Vec<usize>>,
    face_polygons: &HashMap<Vec<usize>, Vec<Vec<usize>>>,
    segments: &HashMap<Vec<usize>, Vec<[Vertex; 2]>>,
    crossing_ids: &HashMap<[usize; 2], Vec<usize>>,
) -> Result<Split, &'static str> {
    let point = |vertex: Vertex| match vertex {
        Vertex::Node(node) => node,
        Vertex::Crossing(edge, ordinal) => crossing_ids[&edge][ordinal],
        Vertex::Feature(..) => panic!(),
    };
    let keyed_faces: Vec<(Vec<usize>, &Vec<usize>)> = faces
        .iter()
        .map(|face| {
            let mut key = face.clone();
            key.sort_unstable();
            (key, face)
        })
        .collect();
    let interior = faces
        .iter()
        .flatten()
        .any(|node| signs[node] == Sign::Inside);
    let mut adjacency = HashMap::<Vertex, Vec<Vertex>>::new();
    keyed_faces.iter().for_each(|(key, _)| {
        if let Some(pairs) = segments.get(key) {
            pairs.iter().for_each(|&[one, two]| {
                adjacency.entry(one).or_default().push(two);
                adjacency.entry(two).or_default().push(one);
            })
        }
    });
    edges.iter().for_each(|&[na, nb]| {
        if signs[&na] == Sign::On && signs[&nb] == Sign::On {
            let mut edge = [na, nb];
            edge.sort_unstable();
            let on_sides: Vec<Sign> = keyed_faces
                .iter()
                .filter(|(_, oriented)| oriented.contains(&na) && oriented.contains(&nb))
                .filter_map(|(key, _)| {
                    face_cuts[key]
                        .on_edges
                        .iter()
                        .find_map(|&(key, side)| (key == edge).then_some(side))
                })
                .collect();
            if on_sides.contains(&Sign::Inside) && on_sides.contains(&Sign::Outside) {
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
        return Ok(if interior {
            Split::Unchanged
        } else {
            Split::Discarded
        });
    }
    if adjacency.values().any(|partners| partners.len() != 2) {
        return Err("open cut chain within a cell");
    }
    let mut polygons = Vec::<Vec<usize>>::new();
    keyed_faces
        .iter()
        .try_for_each(|(key, oriented)| -> Result<(), &'static str> {
            let cut = &face_cuts[key];
            if cut.flush {
                if interior {
                    return Err("refinement required at a face");
                }
            } else {
                let corners = &face_orientations[key];
                let n = corners.len();
                let at = corners
                    .iter()
                    .position(|&node| node == oriented[0])
                    .unwrap();
                let forward = corners[(at + 1) % n] == oriented[1];
                face_polygons[key].iter().for_each(|polygon| {
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
        return Ok(Split::Discarded);
    }
    let nodes: HashSet<usize> = polygons.iter().flatten().copied().collect();
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
    Ok(Split::Cut(CutCell { polygons, clipped }))
}
