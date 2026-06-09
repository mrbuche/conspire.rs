#[cfg(test)]
mod test;

use crate::{
    geometry::mesh::Mesh,
    math::{FxHashMap, FxHashSet},
};

impl<const D: usize> Mesh<D> {
    pub fn boundary_edges(&self) -> Vec<[usize; 2]> {
        self.edge_incidence()
            .into_values()
            .filter_map(|(edge, count)| (count == 1).then_some(edge))
            .collect()
    }
    pub fn boundary_loops(&self) -> Vec<Vec<usize>> {
        let mut next: FxHashMap<usize, usize> = self
            .boundary_edges()
            .into_iter()
            .map(|[a, b]| (a, b))
            .collect();
        let mut loops = Vec::new();
        while let Some(&start) = next.keys().next() {
            let mut nodes = vec![start];
            let mut node = start;
            while let Some(following) = next.remove(&node) {
                if following == start {
                    break;
                }
                nodes.push(following);
                node = following;
            }
            loops.push(nodes);
        }
        loops
    }
    pub fn non_manifold_edges(&self) -> Vec<[usize; 2]> {
        self.edge_incidence()
            .into_values()
            .filter_map(|(edge, count)| (count > 2).then_some(edge))
            .collect()
    }
    pub fn non_manifold_seams(&self) -> Vec<Vec<[usize; 2]>> {
        let edges = self.non_manifold_edges();
        let mut incident = FxHashMap::<usize, Vec<usize>>::default();
        for (e, &[a, b]) in edges.iter().enumerate() {
            incident.entry(a).or_default().push(e);
            incident.entry(b).or_default().push(e);
        }
        let mut visited = vec![false; edges.len()];
        let mut seams = Vec::new();
        for start in 0..edges.len() {
            if visited[start] {
                continue;
            }
            visited[start] = true;
            let mut seam = Vec::new();
            let mut stack = vec![start];
            while let Some(e) = stack.pop() {
                seam.push(edges[e]);
                for node in edges[e] {
                    incident[&node].iter().for_each(|&other| {
                        if !visited[other] {
                            visited[other] = true;
                            stack.push(other);
                        }
                    });
                }
            }
            seams.push(seam);
        }
        seams
    }
    pub fn non_manifold_vertices(&self) -> Vec<usize> {
        let mut faces = Vec::new();
        self.iter().for_each(|block| {
            block
                .iter()
                .for_each(|element| faces.push(element.to_vec()))
        });
        let mut edge_faces = FxHashMap::<[usize; 2], Vec<usize>>::default();
        let mut vertex_faces = FxHashMap::<usize, Vec<usize>>::default();
        for (f, face) in faces.iter().enumerate() {
            for (i, &v) in face.iter().enumerate() {
                vertex_faces.entry(v).or_default().push(f);
                let w = face[(i + 1) % face.len()];
                edge_faces.entry(edge(v, w)).or_default().push(f);
            }
        }
        let mut non_manifold = Vec::new();
        for (&v, incident) in &vertex_faces {
            let local: FxHashMap<usize, usize> =
                incident.iter().enumerate().map(|(i, &f)| (f, i)).collect();
            let mut parent: Vec<usize> = (0..incident.len()).collect();
            for &f in incident {
                let face = &faces[f];
                let (len, pos) = (face.len(), face.iter().position(|&n| n == v).unwrap());
                for x in [face[(pos + len - 1) % len], face[(pos + 1) % len]] {
                    if let Some(shared) = edge_faces.get(&edge(v, x))
                        && shared.len() == 2
                    {
                        union(&mut parent, local[&shared[0]], local[&shared[1]]);
                    }
                }
            }
            let fans: FxHashSet<usize> =
                (0..incident.len()).map(|i| find(&mut parent, i)).collect();
            if fans.len() > 1 {
                non_manifold.push(v);
            }
        }
        non_manifold
    }
    fn edge_incidence(&self) -> FxHashMap<[usize; 2], ([usize; 2], usize)> {
        let mut edges = FxHashMap::default();
        self.iter().for_each(|block| {
            let local_edges = block.local_faces();
            block.iter().for_each(|element| {
                local_edges.iter().for_each(|edge| {
                    let oriented = [element[edge[0]], element[edge[1]]];
                    let key = if oriented[0] < oriented[1] {
                        oriented
                    } else {
                        [oriented[1], oriented[0]]
                    };
                    edges
                        .entry(key)
                        .and_modify(|(_, count)| *count += 1)
                        .or_insert((oriented, 1));
                })
            })
        });
        edges
    }
}

fn edge(a: usize, b: usize) -> [usize; 2] {
    if a < b { [a, b] } else { [b, a] }
}

fn find(parent: &mut [usize], i: usize) -> usize {
    if parent[i] != i {
        parent[i] = find(parent, parent[i]);
    }
    parent[i]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (a, b) = (find(parent, a), find(parent, b));
    parent[a] = b;
}
