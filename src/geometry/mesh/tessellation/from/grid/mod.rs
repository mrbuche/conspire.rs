#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh, tessellation::Tessellation},
    },
    math::{Tensor, TensorVec},
};
use std::{
    array::from_fn,
    collections::{HashMap, HashSet},
};

const DELTAS: [[isize; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

const FACES: [[[usize; 3]; 4]; 6] = [
    [[1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
    [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],
];

const SECTORS: [[isize; 2]; 4] = [[0, 0], [-1, 0], [-1, -1], [0, -1]];
const NEXT: [(usize, bool); 4] = [(0, false), (1, false), (0, true), (1, true)];
const PREV: [(usize, bool); 4] = [(1, false), (0, true), (1, true), (0, false)];

impl<T> From<Voxels<T>> for Tessellation
where
    T: Copy + Default + Ord,
{
    fn from(voxels: Voxels<T>) -> Self {
        let nel = *voxels.nel();
        let [nx, ny, _] = nel;
        let void = T::default();
        let data = voxels.data();
        let cell = |idx: [usize; 3]| idx[0] + nx * idx[1] + nx * ny * idx[2];
        let label = |idx: [isize; 3]| -> T {
            if (0..3).any(|ax| idx[ax] < 0 || idx[ax] >= nel[ax] as isize) {
                void
            } else {
                data[idx[0] as usize + nx * idx[1] as usize + nx * ny * idx[2] as usize]
            }
        };
        let mut quads = Vec::new();
        let mut facemap = HashMap::new();
        let mut edges = HashSet::new();
        for k in 0..nel[2] {
            for j in 0..ny {
                for i in 0..nx {
                    let a = data[cell([i, j, k])];
                    if a == void {
                        continue;
                    }
                    for (face, delta) in DELTAS.iter().enumerate() {
                        let b = label([
                            i as isize + delta[0],
                            j as isize + delta[1],
                            k as isize + delta[2],
                        ]);
                        if a != b {
                            let corners =
                                FACES[face].map(|off| [i + off[0], j + off[1], k + off[2]]);
                            for edge in 0..4 {
                                let (p, q) = (corners[edge], corners[(edge + 1) % 4]);
                                let axis = (0..3).find(|&ax| p[ax] != q[ax]).unwrap();
                                edges.insert((from_fn(|ax| p[ax].min(q[ax])), axis));
                            }
                            facemap.insert((cell([i, j, k]), face), quads.len());
                            quads.push(corners);
                        }
                    }
                }
            }
        }
        let mut parent: Vec<usize> = (0..4 * quads.len()).collect();
        for &(p, t) in &edges {
            let (u, v) = ((t + 1) % 3, (t + 2) % 3);
            let mut labels = [void; 4];
            let mut cells = [0usize; 4];
            for (sector, offset) in SECTORS.iter().enumerate() {
                let mut idx = [0isize; 3];
                idx[t] = p[t] as isize;
                idx[u] = p[u] as isize + offset[0];
                idx[v] = p[v] as isize + offset[1];
                labels[sector] = label(idx);
                if labels[sector] != void {
                    cells[sector] = cell([idx[0] as usize, idx[1] as usize, idx[2] as usize]);
                }
            }
            for start in 0..4 {
                if labels[start] == void || labels[start] == labels[(start + 3) % 4] {
                    continue;
                }
                let mut finish = start;
                while labels[(finish + 1) % 4] == labels[start] && (finish + 1) % 4 != start {
                    finish = (finish + 1) % 4;
                }
                let face =
                    |(local, positive): (usize, bool)| [u, v][local] * 2 + usize::from(!positive);
                let q1 = facemap[&(cells[start], face(PREV[start]))];
                let q2 = facemap[&(cells[finish], face(NEXT[finish]))];
                for offset in [0, 1] {
                    let mut endpoint = p;
                    endpoint[t] += offset;
                    let a = 4 * q1 + local(&quads[q1], endpoint);
                    let b = 4 * q2 + local(&quads[q2], endpoint);
                    union(&mut parent, a, b);
                }
            }
        }
        let mut coordinates = Coordinates::new();
        let mut remap = HashMap::new();
        let mut triangles = Vec::with_capacity(2 * quads.len());
        for (q, corners) in quads.iter().enumerate() {
            let ids: [usize; 4] = from_fn(|l| {
                let root = find(&mut parent, 4 * q + l);
                *remap.entry(root).or_insert_with(|| {
                    let id = coordinates.len();
                    coordinates.push(Coordinate::const_from(from_fn(|ax| corners[l][ax] as f64)));
                    id
                })
            });
            triangles.push([ids[0], ids[1], ids[2]]);
            triangles.push([ids[0], ids[2], ids[3]]);
        }
        let triangles = resolve_pinches(&triangles, &mut coordinates);
        let connectivities = vec![Connectivity::Triangular(triangles.into())];
        Tessellation::from(Mesh::from((connectivities, coordinates)))
    }
}

fn resolve_pinches(triangles: &[[usize; 3]], coordinates: &mut Coordinates<3>) -> Vec<[usize; 3]> {
    let mut vertex_faces: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut edge_faces: HashMap<[usize; 2], Vec<usize>> = HashMap::new();
    for (f, tri) in triangles.iter().enumerate() {
        for i in 0..3 {
            let (v, w) = (tri[i], tri[(i + 1) % 3]);
            vertex_faces.entry(v).or_default().push(f);
            edge_faces.entry(edge(v, w)).or_default().push(f);
        }
    }
    let mut assignment = triangles.to_vec();
    for (&v, incident) in &vertex_faces {
        let index: HashMap<usize, usize> =
            incident.iter().enumerate().map(|(i, &f)| (f, i)).collect();
        let mut parent: Vec<usize> = (0..incident.len()).collect();
        for &f in incident {
            let tri = &triangles[f];
            let pos = tri.iter().position(|&n| n == v).unwrap();
            for x in [tri[(pos + 2) % 3], tri[(pos + 1) % 3]] {
                if let Some(shared) = edge_faces.get(&edge(v, x))
                    && shared.len() == 2
                {
                    union(&mut parent, index[&shared[0]], index[&shared[1]]);
                }
            }
        }
        let mut fan: HashMap<usize, usize> = HashMap::new();
        for i in 0..incident.len() {
            let root = find(&mut parent, i);
            let next = fan.len();
            fan.entry(root).or_insert(next);
        }
        if fan.len() < 2 {
            continue;
        }
        let mut copies: HashMap<usize, usize> = HashMap::new();
        for (i, &f) in incident.iter().enumerate() {
            let root = find(&mut parent, i);
            if fan[&root] == 0 {
                continue;
            }
            let id = *copies.entry(root).or_insert_with(|| {
                let point = Coordinate::const_from(from_fn(|ax| coordinates[v][ax]));
                let id = coordinates.len();
                coordinates.push(point);
                id
            });
            let pos = triangles[f].iter().position(|&n| n == v).unwrap();
            assignment[f][pos] = id;
        }
    }
    assignment
}

fn edge(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn local(corners: &[[usize; 3]; 4], corner: [usize; 3]) -> usize {
    corners.iter().position(|&c| c == corner).unwrap()
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[ra] = rb;
    }
}
