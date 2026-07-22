#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{Connectivity, Mesh},
    },
    math::{FxHashMap, FxHashSet, Scalar, Tensor, TensorVec},
};
use std::array::from_fn;

const N: usize = 3;
const SPLIT_ABOVE: Scalar = 4.0 / 3.0;
const COLLAPSE_BELOW: Scalar = 4.0 / 5.0;

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum Label {
    Free,
    Curve(usize),
    Corner,
}

pub(crate) struct Constraints<const D: usize> {
    pub labels: Vec<Label>,
    pub edges: FxHashMap<(usize, usize), usize>,
    pub curves: Vec<Vec<Coordinate<D>>>,
}

pub(crate) fn remesh<const D: usize, F>(
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    iterations: usize,
    mut sizing_of: F,
    mut constraints: Option<&mut Constraints<D>>,
) -> Result<(), &'static str>
where
    F: FnMut(&[[usize; N]], &Coordinates<D>, &FxHashMap<(usize, usize), Scalar>) -> Vec<Scalar>,
{
    let surface = (D == 3).then(|| {
        let coordinates: &Coordinates<3> =
            unsafe { &*(&*coordinates as *const Coordinates<D>).cast() };
        Surface::new(connectivity, coordinates)
    });
    for _ in 0..iterations {
        let lengths = edge_lengths(connectivity, coordinates);
        if lengths.is_empty() {
            break;
        }
        let mut sizing = sizing_of(connectivity, coordinates, &lengths);
        split_long_edges(
            connectivity,
            coordinates,
            &lengths,
            &mut sizing,
            constraints.as_deref_mut(),
        );
        let lengths = edge_lengths(connectivity, coordinates);
        collapse_short_edges(
            connectivity,
            coordinates,
            &lengths,
            &mut sizing,
            constraints.as_deref_mut(),
        );
        flip_edges(connectivity, coordinates, constraints.as_deref());
        tangential_smooth(connectivity, coordinates, constraints.as_deref());
        if let Some(surface) = &surface {
            let coordinates: &mut Coordinates<3> =
                unsafe { &mut *(coordinates as *mut Coordinates<D>).cast() };
            match constraints.as_deref() {
                None => surface.reproject(coordinates),
                Some(constraints) => {
                    let constraints: &Constraints<3> =
                        unsafe { &*(constraints as *const Constraints<D>).cast() };
                    surface.reproject_constrained(coordinates, constraints)
                }
            }
        }
    }
    Ok(())
}

fn polyline_closest<const D: usize>(
    point: &Coordinate<D>,
    polyline: &[Coordinate<D>],
) -> Coordinate<D> {
    (0..polyline.len() - 1)
        .map(|i| {
            let ab = &polyline[i + 1] - &polyline[i];
            let ap = point - &polyline[i];
            let denominator = &ab * &ab;
            let closest = if denominator == 0.0 {
                polyline[i].clone()
            } else {
                let t = ((&ap * &ab) / denominator).clamp(0.0, 1.0);
                &polyline[i] + &(ab * t)
            };
            ((&closest - point).norm(), closest)
        })
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .map(|(_, closest)| closest)
        .unwrap_or_else(|| polyline[0].clone())
}

struct Surface {
    mesh: Mesh<3>,
    bvh: BoundingVolumeHierarchy<3>,
}

impl Surface {
    fn new(connectivity: &[[usize; N]], coordinates: &Coordinates<3>) -> Self {
        let mesh = Mesh::from((
            vec![Connectivity::Triangular(connectivity.to_vec().into())],
            coordinates.clone(),
        ));
        let bvh = BoundingVolumeHierarchy::from(&mesh);
        Self { mesh, bvh }
    }
    fn reproject(&self, coordinates: &mut Coordinates<3>) {
        let elements: Vec<&[usize]> = self.mesh.connectivities().iter().flatten().collect();
        let surface = self.mesh.coordinates();
        *coordinates = (0..coordinates.len())
            .map(|vertex| {
                self.bvh
                    .closest_point(&coordinates[vertex], surface, &elements)
                    .map_or_else(|| coordinates[vertex].clone(), |(point, _)| point)
            })
            .collect();
    }
    fn reproject_constrained(
        &self,
        coordinates: &mut Coordinates<3>,
        constraints: &Constraints<3>,
    ) {
        let elements: Vec<&[usize]> = self.mesh.connectivities().iter().flatten().collect();
        let surface = self.mesh.coordinates();
        *coordinates = (0..coordinates.len())
            .map(|vertex| match constraints.labels[vertex] {
                Label::Corner => coordinates[vertex].clone(),
                Label::Curve(curve) => {
                    polyline_closest(&coordinates[vertex], &constraints.curves[curve])
                }
                Label::Free => self
                    .bvh
                    .closest_point(&coordinates[vertex], surface, &elements)
                    .map_or_else(|| coordinates[vertex].clone(), |(point, _)| point),
            })
            .collect();
    }
}

fn edge(a: usize, b: usize) -> (usize, usize) {
    if a < b { (a, b) } else { (b, a) }
}

fn edge_lengths<const D: usize>(
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
) -> FxHashMap<(usize, usize), Scalar> {
    let mut lengths = FxHashMap::default();
    connectivity.iter().for_each(|&[a, b, c]| {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            lengths
                .entry(edge(u, v))
                .or_insert_with(|| (&coordinates[v] - &coordinates[u]).norm());
        }
    });
    lengths
}

fn split_long_edges<const D: usize>(
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    lengths: &FxHashMap<(usize, usize), Scalar>,
    sizing: &mut Vec<Scalar>,
    mut constraints: Option<&mut Constraints<D>>,
) {
    let mut midpoints: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for (&(u, v), &length) in lengths {
        if length > SPLIT_ABOVE * 0.5 * (sizing[u] + sizing[v]) {
            let midpoint = &(&coordinates[u] + &coordinates[v]) * 0.5;
            midpoints.insert((u, v), coordinates.len());
            coordinates.push(midpoint);
            sizing.push(0.5 * (sizing[u] + sizing[v]));
            if let Some(constraints) = constraints.as_mut() {
                match constraints.edges.get(&(u, v)).copied() {
                    Some(curve) => constraints.labels.push(Label::Curve(curve)),
                    None => constraints.labels.push(Label::Free),
                }
            }
        }
    }
    if midpoints.is_empty() {
        return;
    }
    if let Some(constraints) = constraints {
        midpoints.iter().for_each(|(&(u, v), &midpoint)| {
            if let Some(curve) = constraints.edges.remove(&(u, v)) {
                constraints.edges.insert(edge(u, midpoint), curve);
                constraints.edges.insert(edge(midpoint, v), curve);
            }
        });
    }
    let mut split = Vec::with_capacity(connectivity.len());
    for &[a, b, c] in connectivity.iter() {
        let ab = midpoints.get(&edge(a, b)).copied();
        let bc = midpoints.get(&edge(b, c)).copied();
        let ca = midpoints.get(&edge(c, a)).copied();
        match (ab, bc, ca) {
            (None, None, None) => split.push([a, b, c]),
            (Some(p), None, None) => split.extend([[a, p, c], [p, b, c]]),
            (None, Some(q), None) => split.extend([[b, q, a], [q, c, a]]),
            (None, None, Some(r)) => split.extend([[c, r, b], [r, a, b]]),
            (Some(p), Some(q), None) => split.extend([[a, p, q], [a, q, c], [p, b, q]]),
            (None, Some(q), Some(r)) => split.extend([[b, q, r], [b, r, a], [q, c, r]]),
            (Some(p), None, Some(r)) => split.extend([[c, r, p], [c, p, b], [r, a, p]]),
            (Some(p), Some(q), Some(r)) => {
                split.extend([[a, p, r], [p, b, q], [r, q, c], [p, q, r]])
            }
        }
    }
    *connectivity = split;
}

fn collapse_short_edges<const D: usize>(
    connectivity: &mut Vec<[usize; N]>,
    coordinates: &mut Coordinates<D>,
    lengths: &FxHashMap<(usize, usize), Scalar>,
    sizing: &mut Vec<Scalar>,
    constraints: Option<&mut Constraints<D>>,
) {
    let vertices = coordinates.len();
    let mut neighbors: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); vertices];
    let mut vertex_faces: Vec<Vec<usize>> = vec![Vec::new(); vertices];
    let mut edge_faces: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (face, &[a, b, c]) in connectivity.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            neighbors[u].insert(v);
            neighbors[v].insert(u);
            edge_faces.entry(edge(u, v)).or_default().push(face);
        }
        for w in [a, b, c] {
            vertex_faces[w].push(face);
        }
    }
    let mut boundary = vec![false; vertices];
    for (&(u, v), faces) in &edge_faces {
        if faces.len() == 1 {
            boundary[u] = true;
            boundary[v] = true;
        }
    }
    let mut constrained_neighbors: FxHashMap<usize, FxHashSet<usize>> = FxHashMap::default();
    if let Some(constraints) = constraints.as_ref() {
        constraints.edges.keys().for_each(|&(a, b)| {
            constrained_neighbors.entry(a).or_default().insert(b);
            constrained_neighbors.entry(b).or_default().insert(a);
        });
    }
    let mut short: Vec<(usize, usize)> = lengths
        .iter()
        .filter(|&(&(u, v), &length)| length < COLLAPSE_BELOW * 0.5 * (sizing[u] + sizing[v]))
        .map(|(&e, _)| e)
        .collect();
    short.sort_by(|a, b| lengths[a].total_cmp(&lengths[b]));
    let mut merge: Vec<usize> = (0..vertices).collect();
    let mut positions: FxHashMap<usize, Coordinate<D>> = FxHashMap::default();
    let mut touched = vec![false; vertices];
    for (u, v) in short {
        if touched[u] || touched[v] {
            continue;
        }
        let opposites: FxHashSet<usize> = edge_faces[&edge(u, v)]
            .iter()
            .map(|&face| {
                connectivity[face]
                    .into_iter()
                    .find(|&x| x != u && x != v)
                    .unwrap()
            })
            .collect();
        if &neighbors[u] & &neighbors[v] != opposites {
            continue;
        }
        if let (Some(nu), Some(nv)) = (constrained_neighbors.get(&u), constrained_neighbors.get(&v))
            && nu.iter().any(|&w| w != v && nv.contains(&w))
        {
            continue;
        }
        let (survivor, removed, position) = match constraints.as_ref() {
            Some(constraints) => match (constraints.labels[u], constraints.labels[v]) {
                (Label::Free, Label::Free) => match (boundary[u], boundary[v]) {
                    (false, false) => (u, v, &(&coordinates[u] + &coordinates[v]) * 0.5),
                    (true, false) => (u, v, coordinates[u].clone()),
                    (false, true) => (v, u, coordinates[v].clone()),
                    (true, true) if edge_faces[&edge(u, v)].len() == 1 => {
                        (u, v, &(&coordinates[u] + &coordinates[v]) * 0.5)
                    }
                    (true, true) => continue,
                },
                (_, Label::Free) => (u, v, coordinates[u].clone()),
                (Label::Free, _) => (v, u, coordinates[v].clone()),
                _ if !constraints.edges.contains_key(&edge(u, v)) => continue,
                (Label::Corner, Label::Corner) => continue,
                (Label::Corner, _) => (u, v, coordinates[u].clone()),
                (_, Label::Corner) => (v, u, coordinates[v].clone()),
                _ => (u, v, &(&coordinates[u] + &coordinates[v]) * 0.5),
            },
            None => match (boundary[u], boundary[v]) {
                (false, false) => (u, v, &(&coordinates[u] + &coordinates[v]) * 0.5),
                (true, false) => (u, v, coordinates[u].clone()),
                (false, true) => (v, u, coordinates[v].clone()),
                (true, true) if edge_faces[&edge(u, v)].len() == 1 => {
                    (u, v, &(&coordinates[u] + &coordinates[v]) * 0.5)
                }
                (true, true) => continue,
            },
        };
        if neighbors[u]
            .iter()
            .chain(&neighbors[v])
            .filter(|&&w| w != u && w != v)
            .any(|&w| {
                (&coordinates[w] - &position).norm()
                    > SPLIT_ABOVE * 0.5 * (sizing[survivor] + sizing[w])
            })
        {
            continue;
        }
        if D == 3 {
            let moved = |w: usize| {
                if w == u || w == v {
                    &position
                } else {
                    &coordinates[w]
                }
            };
            let folds = vertex_faces[u].iter().chain(&vertex_faces[v]).any(|&face| {
                let [a, b, c] = connectivity[face];
                let degenerate = [a, b, c].contains(&u) && [a, b, c].contains(&v);
                !degenerate
                    && triangle_normal(&coordinates[a], &coordinates[b], &coordinates[c])
                        * triangle_normal(moved(a), moved(b), moved(c))
                        <= 0.0
            });
            if folds {
                continue;
            }
        }
        merge[removed] = survivor;
        positions.insert(survivor, position);
        neighbors[u]
            .iter()
            .chain(&neighbors[v])
            .chain([&u, &v])
            .for_each(|&w| touched[w] = true);
    }
    let mut kept: Vec<[usize; N]> = Vec::with_capacity(connectivity.len());
    for &[a, b, c] in connectivity.iter() {
        let face = [merge[a], merge[b], merge[c]];
        if face[0] != face[1] && face[1] != face[2] && face[2] != face[0] {
            kept.push(face);
        }
    }
    let mut used = vec![false; vertices];
    kept.iter()
        .flatten()
        .for_each(|&vertex| used[vertex] = true);
    let mut remap = vec![usize::MAX; vertices];
    let mut compact = Coordinates::new();
    let mut compact_sizing = Vec::new();
    for vertex in 0..vertices {
        if used[vertex] {
            remap[vertex] = compact.len();
            compact.push(
                positions
                    .remove(&vertex)
                    .unwrap_or_else(|| coordinates[vertex].clone()),
            );
            compact_sizing.push(sizing[vertex]);
        }
    }
    kept.iter_mut()
        .for_each(|face| face.iter_mut().for_each(|vertex| *vertex = remap[*vertex]));
    if let Some(constraints) = constraints {
        let mut labels = vec![Label::Free; compact.len()];
        (0..vertices).for_each(|vertex| {
            if used[vertex] {
                labels[remap[vertex]] = constraints.labels[vertex];
            }
        });
        constraints.labels = labels;
        let mut edges = FxHashMap::default();
        constraints.edges.iter().for_each(|(&(a, b), &curve)| {
            let (a, b) = (merge[a], merge[b]);
            if a != b && used[a] && used[b] {
                edges.insert(edge(remap[a], remap[b]), curve);
            }
        });
        constraints.edges = edges;
    }
    *connectivity = kept;
    *coordinates = compact;
    *sizing = compact_sizing;
}

fn opposite_and_direction(face: &[usize; N], u: usize, v: usize) -> (usize, bool) {
    for i in 0..N {
        let (a, b) = (face[i], face[(i + 1) % N]);
        if a == u && b == v {
            return (face[(i + 2) % N], true);
        } else if a == v && b == u {
            return (face[(i + 2) % N], false);
        }
    }
    unreachable!()
}

fn deviation(valence: i64, target: i64) -> i64 {
    (valence - target) * (valence - target)
}

fn triangle_normal<const D: usize>(
    a: &Coordinate<D>,
    b: &Coordinate<D>,
    c: &Coordinate<D>,
) -> Coordinate<3> {
    let (e, f) = (b - a, c - a);
    Coordinate::const_from([
        e[1] * f[2] - e[2] * f[1],
        e[2] * f[0] - e[0] * f[2],
        e[0] * f[1] - e[1] * f[0],
    ])
}

fn flip_edges<const D: usize>(
    connectivity: &mut [[usize; N]],
    coordinates: &Coordinates<D>,
    constraints: Option<&Constraints<D>>,
) {
    let vertices = connectivity
        .iter()
        .flatten()
        .copied()
        .max()
        .map_or(0, |m| m + 1);
    let mut edge_faces: FxHashMap<(usize, usize), Vec<usize>> = FxHashMap::default();
    for (face, &[a, b, c]) in connectivity.iter().enumerate() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            edge_faces.entry(edge(u, v)).or_default().push(face);
        }
    }
    let mut valence = vec![0_i64; vertices];
    let mut boundary = vec![false; vertices];
    for (&(u, v), faces) in &edge_faces {
        valence[u] += 1;
        valence[v] += 1;
        if faces.len() == 1 {
            boundary[u] = true;
            boundary[v] = true;
        }
    }
    let target = |vertex: usize| if boundary[vertex] { 4 } else { 6 };
    let interior: Vec<(usize, usize)> = edge_faces
        .iter()
        .filter(|(_, faces)| faces.len() == 2)
        .map(|(&e, _)| e)
        .collect();
    let mut touched = vec![false; vertices];
    for (u, v) in interior {
        if touched[u] || touched[v] {
            continue;
        }
        if constraints.is_some_and(|constraints| constraints.edges.contains_key(&edge(u, v))) {
            continue;
        }
        let faces = &edge_faces[&edge(u, v)];
        let (left, right) = (faces[0], faces[1]);
        let (c, c_directed) = opposite_and_direction(&connectivity[left], u, v);
        let (d, _) = opposite_and_direction(&connectivity[right], u, v);
        if c == d || touched[c] || touched[d] || edge_faces.contains_key(&edge(c, d)) {
            continue;
        }
        let before = deviation(valence[u], target(u))
            + deviation(valence[v], target(v))
            + deviation(valence[c], target(c))
            + deviation(valence[d], target(d));
        let after = deviation(valence[u] - 1, target(u))
            + deviation(valence[v] - 1, target(v))
            + deviation(valence[c] + 1, target(c))
            + deviation(valence[d] + 1, target(d));
        if after >= before {
            continue;
        }
        let (uv, vu) = if c_directed { (c, d) } else { (d, c) };
        if D == 3 {
            let face_normal = |[a, b, c]: [usize; N]| {
                triangle_normal(&coordinates[a], &coordinates[b], &coordinates[c])
            };
            let (l, r) = (
                face_normal(connectivity[left]),
                face_normal(connectivity[right]),
            );
            let surface = Coordinate::const_from([l[0] + r[0], l[1] + r[1], l[2] + r[2]]);
            if face_normal([v, uv, vu]) * &surface <= 0.0
                || face_normal([uv, u, vu]) * &surface <= 0.0
            {
                continue;
            }
        }
        connectivity[left] = [v, uv, vu];
        connectivity[right] = [uv, u, vu];
        valence[u] -= 1;
        valence[v] -= 1;
        valence[c] += 1;
        valence[d] += 1;
        [u, v, c, d]
            .into_iter()
            .for_each(|vertex| touched[vertex] = true);
    }
}

fn tangential_smooth<const D: usize>(
    connectivity: &[[usize; N]],
    coordinates: &mut Coordinates<D>,
    constraints: Option<&Constraints<D>>,
) {
    let vertices = coordinates.len();
    let mut neighbors: Vec<FxHashSet<usize>> = vec![FxHashSet::default(); vertices];
    let mut edge_uses: FxHashMap<(usize, usize), usize> = FxHashMap::default();
    for &[a, b, c] in connectivity.iter() {
        for (u, v) in [(a, b), (b, c), (c, a)] {
            neighbors[u].insert(v);
            neighbors[v].insert(u);
            *edge_uses.entry(edge(u, v)).or_default() += 1;
        }
    }
    let mut boundary = vec![false; vertices];
    for (&(u, v), &uses) in &edge_uses {
        if uses == 1 {
            boundary[u] = true;
            boundary[v] = true;
        }
    }
    let mut normals = vec![[0.0; 3]; vertices];
    if D == 3 {
        for &[a, b, c] in connectivity.iter() {
            let (e, f) = (
                &coordinates[b] - &coordinates[a],
                &coordinates[c] - &coordinates[a],
            );
            let face = [
                e[1] * f[2] - e[2] * f[1],
                e[2] * f[0] - e[0] * f[2],
                e[0] * f[1] - e[1] * f[0],
            ];
            for vertex in [a, b, c] {
                (0..3).for_each(|i| normals[vertex][i] += face[i]);
            }
        }
        normals.iter_mut().for_each(|normal| {
            let length = (normal[0].powi(2) + normal[1].powi(2) + normal[2].powi(2)).sqrt();
            if length > 0.0 {
                (0..3).for_each(|i| normal[i] /= length);
            }
        });
    }
    let mut curve_neighbors: Vec<Vec<usize>> = vec![Vec::new(); vertices];
    if let Some(constraints) = constraints {
        constraints.edges.keys().for_each(|&(u, v)| {
            curve_neighbors[u].push(v);
            curve_neighbors[v].push(u);
        });
    }
    let mut smoothed = Coordinates::new();
    for vertex in 0..vertices {
        if let Some(constraints) = constraints {
            match constraints.labels[vertex] {
                Label::Corner => {
                    smoothed.push(coordinates[vertex].clone());
                    continue;
                }
                Label::Curve(_) => {
                    if let [a, b] = curve_neighbors[vertex][..] {
                        smoothed.push(&(&coordinates[a] + &coordinates[b]) * 0.5);
                    } else {
                        smoothed.push(coordinates[vertex].clone());
                    }
                    continue;
                }
                Label::Free => {}
            }
        }
        if boundary[vertex] || neighbors[vertex].is_empty() {
            smoothed.push(coordinates[vertex].clone());
            continue;
        }
        let degree = neighbors[vertex].len() as Scalar;
        let centroid = neighbors[vertex]
            .iter()
            .map(|&neighbor| &coordinates[neighbor])
            .sum::<Coordinate<D>>()
            / degree;
        let mut displacement = &centroid - &coordinates[vertex];
        if D == 3 {
            let normal = &normals[vertex];
            let along = (0..3).map(|i| displacement[i] * normal[i]).sum::<Scalar>();
            let tangential: [Scalar; D] = from_fn(|i| displacement[i] - along * normal[i]);
            displacement = tangential.into();
        }
        smoothed.push(&coordinates[vertex] + &displacement);
    }
    *coordinates = smoothed;
}
