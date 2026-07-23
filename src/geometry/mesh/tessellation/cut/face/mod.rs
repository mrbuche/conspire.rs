use super::{Sign, Vertex};
use crate::geometry::{Coordinate, mesh::tessellation::D};
use std::{array::from_fn, collections::HashMap, mem::take};

pub(super) struct FaceCut {
    pub(super) endpoints: Vec<Vertex>,
    pub(super) sides: Vec<Sign>,
    pub(super) interiors: Vec<Vec<usize>>,
    pub(super) emitted: Vec<usize>,
    pub(super) on_edges: Vec<([usize; 2], Sign)>,
    pub(super) inside: bool,
    pub(super) flush: bool,
}

pub(super) fn face_cut(
    corners: &[usize; 4],
    signs: &HashMap<usize, Sign>,
    crossings: &HashMap<[usize; 2], Vec<Coordinate<D>>>,
) -> Result<FaceCut, &'static str> {
    let statuses: [Sign; 4] = from_fn(|i| signs[&corners[i]]);
    let edge_keys: [[usize; 2]; 4] = from_fn(|i| {
        let mut key = [corners[i], corners[(i + 1) % 4]];
        key.sort_unstable();
        key
    });
    let counts: [usize; 4] = from_fn(|i| crossings.get(&edge_keys[i]).map_or(0, Vec::len));
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
        let edge_flips: usize = counts[from] + ons.iter().map(|&o| counts[o]).sum::<usize>();
        let needs_pass = (edge_flips % 2 == 1) != change;
        if ons.is_empty() {
            if needs_pass {
                return Err("inconsistent signs around a face");
            }
        } else if needs_pass {
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
    let endpoint = |vertex: Vertex,
                    side: &mut Sign,
                    current: &mut Vec<usize>,
                    endpoints: &mut Vec<Vertex>,
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
        endpoints.push(vertex);
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
                        Vertex::Node(corners[at]),
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
        let key = edge_keys[at];
        let forward = corners[at] == key[0];
        (0..counts[at]).for_each(|i| {
            let ordinal = if forward { i } else { counts[at] - 1 - i };
            endpoint(
                Vertex::Crossing(key, ordinal),
                &mut side,
                &mut current,
                &mut endpoints,
                &mut sides,
                &mut interiors,
                &mut prefix,
                &mut opened,
            );
        });
        if counts[at] == 0 && statuses[at] == Sign::On && statuses[(at + 1) % 4] == Sign::On {
            on_edges.push((key, side));
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

pub(super) fn clip_face(
    cut: &FaceCut,
    chords: Option<&Vec<[Vertex; 2]>>,
    crossing_ids: &HashMap<[usize; 2], Vec<usize>>,
) -> Vec<Vec<usize>> {
    let point = |vertex: Vertex| match vertex {
        Vertex::Node(node) => node,
        Vertex::Crossing(edge, ordinal) => crossing_ids[&edge][ordinal],
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
    let arcs: HashMap<Vertex, usize> = cut
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
