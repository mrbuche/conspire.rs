#[cfg(test)]
mod test;

use std::collections::HashMap;

pub(super) fn sphere(faces: &[Vec<usize>]) -> Result<(), String> {
    let mut edges = HashMap::<(usize, usize), usize>::new();
    let mut corners = HashMap::<usize, Vec<(usize, usize)>>::new();
    for (index, face) in faces.iter().enumerate() {
        let length = face.len();
        for spot in 0..length {
            let (previous, node, next) = (
                face[(spot + length - 1) % length],
                face[spot],
                face[(spot + 1) % length],
            );
            if edges.insert((node, next), index).is_some() {
                return Err("an edge is used twice in the same direction".to_string());
            }
            corners.entry(node).or_default().push((previous, next));
        }
    }
    if edges.keys().any(|&(a, b)| !edges.contains_key(&(b, a))) {
        return Err("the surface is not closed".to_string());
    }
    for (node, node_corners) in &corners {
        let successors = node_corners.iter().copied().collect::<HashMap<_, _>>();
        let (first, mut current) = node_corners[0];
        let mut count = 1;
        while current != first {
            current = *successors
                .get(&current)
                .ok_or_else(|| format!("node {node} is not on a closed fan"))?;
            count += 1;
        }
        if successors.len() != node_corners.len() || count != node_corners.len() {
            return Err(format!("node {node} is pinched"));
        }
    }
    let mut roots = (0..faces.len()).collect::<Vec<_>>();
    edges.iter().for_each(|(&(a, b), &face)| {
        let (x, y) = (find(&mut roots, face), find(&mut roots, edges[&(b, a)]));
        roots[x] = y
    });
    let root = find(&mut roots, 0);
    if (1..faces.len()).any(|face| find(&mut roots, face) != root) {
        return Err("the surface has several components".to_string());
    }
    if corners.len() + faces.len() != edges.len() / 2 + 2 {
        return Err("the surface is not a sphere".to_string());
    }
    Ok(())
}

fn find(roots: &mut [usize], mut node: usize) -> usize {
    while roots[node] != node {
        roots[node] = roots[roots[node]];
        node = roots[node]
    }
    node
}
