#[cfg(test)]
mod test;

use super::{Sign, Vertex};
use crate::geometry::{Coordinate, mesh::tessellation::D};
use std::{collections::HashMap, mem::take};

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
    corners: &[usize],
    signs: &HashMap<usize, Sign>,
    crossings: &HashMap<[usize; 2], Vec<Coordinate<D>>>,
) -> Result<FaceCut, &'static str> {
    let n = corners.len();
    let statuses: Vec<Sign> = corners.iter().map(|node| signs[node]).collect();
    let edge_keys: Vec<[usize; 2]> = (0..n)
        .map(|i| {
            let mut key = [corners[i], corners[(i + 1) % n]];
            key.sort_unstable();
            key
        })
        .collect();
    let counts: Vec<usize> = edge_keys
        .iter()
        .map(|key| crossings.get(key).map_or(0, Vec::len))
        .collect();
    let flip = |sign| {
        if sign == Sign::Inside {
            Sign::Outside
        } else {
            Sign::Inside
        }
    };
    let decisive: Vec<usize> = (0..n).filter(|&i| statuses[i] != Sign::On).collect();
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
    let mut pass = vec![false; n];
    for (w, &from) in decisive.iter().enumerate() {
        let to = decisive[(w + 1) % decisive.len()];
        let mut ons = Vec::new();
        let mut at = (from + 1) % n;
        while at != to {
            ons.push(at);
            at = (at + 1) % n;
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
    for step in 0..n {
        let at = (start + step) % n;
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
        if counts[at] == 0 && statuses[at] == Sign::On && statuses[(at + 1) % n] == Sign::On {
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
    vias: Option<&Vec<Vec<Vertex>>>,
    crossing_ids: &HashMap<[usize; 2], Vec<usize>>,
    feature_ids: &HashMap<([usize; 4], usize), usize>,
) -> Vec<Vec<usize>> {
    let point = |vertex: Vertex| match vertex {
        Vertex::Node(node) => node,
        Vertex::Crossing(edge, ordinal) => crossing_ids[&edge][ordinal],
        Vertex::Feature(face, crease) => feature_ids[&(face, crease)],
        Vertex::Corner(_) => panic!(),
    };
    if cut.endpoints.is_empty() {
        return if cut.inside && cut.emitted.len() > 2 {
            vec![cut.emitted.clone()]
        } else {
            vec![]
        };
    }
    let mut partner = HashMap::new();
    let mut along = HashMap::<[Vertex; 2], Vec<Vertex>>::new();
    chords
        .unwrap()
        .iter()
        .enumerate()
        .for_each(|(chord, &[one, two])| {
            partner.insert(one, two);
            partner.insert(two, one);
            let riding = vias
                .and_then(|vias| vias.get(chord))
                .cloned()
                .unwrap_or_default();
            let mut back = riding.clone();
            back.reverse();
            along.insert([one, two], riding);
            along.insert([two, one], back);
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
                let next = partner[&end];
                along[&[end, next]]
                    .iter()
                    .for_each(|&riding| polygon.push(point(riding)));
                let jump = arcs[&next];
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
