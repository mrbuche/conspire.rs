#[cfg(test)]
mod test;

use super::find;
use std::collections::HashMap;

/// The topology of faces that make up closed, consistently oriented surfaces.
pub struct Surface {
    euler_characteristics: Vec<isize>,
}

impl Surface {
    pub fn number_of_components(&self) -> usize {
        self.euler_characteristics.len()
    }
    /// The Euler characteristic of each component.
    pub fn euler_characteristics(&self) -> &[isize] {
        &self.euler_characteristics
    }
    pub fn euler_characteristic(&self) -> isize {
        self.euler_characteristics.iter().sum()
    }
    /// The number of handles of each component.
    pub fn genera(&self) -> Vec<usize> {
        self.euler_characteristics
            .iter()
            .map(|characteristic| ((2 - characteristic) / 2) as usize)
            .collect()
    }
    /// Whether the surface is one topological sphere.
    pub fn is_sphere(&self) -> bool {
        self.euler_characteristics == [2]
    }
    /// Ok if the surface is one topological sphere, else why not.
    pub fn ensure_sphere(&self) -> Result<(), String> {
        match self.euler_characteristics() {
            [2] => Ok(()),
            [characteristic] => Err(format!(
                "the surface is not a sphere, it has genus {}",
                (2 - characteristic) / 2
            )),
            components => Err(format!(
                "the surface has {} components, not one",
                components.len()
            )),
        }
    }
}

impl<F: AsRef<[usize]>> TryFrom<&[F]> for Surface {
    type Error = String;
    fn try_from(faces: &[F]) -> Result<Self, String> {
        if faces.is_empty() {
            return Err("there are no faces".to_string());
        }
        if faces.iter().any(|face| face.as_ref().len() < 3) {
            return Err("a face has fewer than three nodes".to_string());
        }
        let mut edges = HashMap::<(usize, usize), usize>::new();
        let mut corners = HashMap::<usize, Vec<(usize, usize)>>::new();
        let mut nodes_faces = HashMap::<usize, usize>::new();
        for (index, face) in faces.iter().enumerate() {
            let face = face.as_ref();
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
                nodes_faces.insert(node, index);
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
        let mut components = HashMap::new();
        let faces_components = (0..faces.len())
            .map(|face| {
                let root = find(&mut roots, face);
                let next = components.len();
                *components.entry(root).or_insert(next)
            })
            .collect::<Vec<_>>();
        let mut counts = vec![[0isize; 3]; components.len()];
        nodes_faces
            .values()
            .for_each(|&face| counts[faces_components[face]][0] += 1);
        edges
            .iter()
            .filter(|((a, b), _)| a < b)
            .for_each(|(_, &face)| counts[faces_components[face]][1] += 1);
        faces_components
            .iter()
            .for_each(|&component| counts[component][2] += 1);
        Ok(Self {
            euler_characteristics: counts
                .iter()
                .map(|[nodes, edges, faces]| nodes - edges + faces)
                .collect(),
        })
    }
}
