#[cfg(test)]
mod test;

use crate::{geometry::mesh::Mesh, math::FxHashMap};

/// What a node of a mesh of several materials sits on.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NodeClass {
    /// Inside a volume, on no interface.
    Interior,
    /// On one surface, which separates two volumes or a volume from the outside.
    Surface,
    /// On a curve, where two edges leave it along more than one surface.
    Curve,
    /// Where more or fewer than two such edges meet.
    Vertex,
}

const OUTSIDE: usize = usize::MAX;

type Surface = [usize; 2];

fn root(parents: &mut [usize], mut element: usize) -> usize {
    while parents[element] != element {
        parents[element] = parents[parents[element]];
        element = parents[element];
    }
    element
}

fn add(surfaces: &mut Vec<Surface>, surface: Surface) {
    if !surfaces.contains(&surface) {
        surfaces.push(surface);
    }
}

impl Mesh<3> {
    /// The class of every node, from the materials of the blocks the elements belong to.
    ///
    /// A volume is a set of elements of one material joined by faces, and a surface is the
    /// faces between the same two volumes, or between a volume and the outside. A node on no
    /// surface is interior, on one is a surface node, and on several is on a curve when exactly
    /// two edges that lie on more than one surface leave it and is a vertex otherwise. The faces
    /// are the mesh's own, so a sharp feature is not told apart from the flat surface around it.
    pub fn classify_nodes(&self) -> Result<Vec<NodeClass>, &'static str> {
        let labels = self.block_labels();
        let mut elements = Vec::new();
        for (block, connectivity) in self.iter().enumerate() {
            let material = labels.map_or(block, |labels| labels[block]);
            for element in connectivity.iter() {
                elements.push((material, connectivity.element_faces(element)));
            }
        }
        let mut faces = FxHashMap::<Vec<usize>, (Vec<usize>, Vec<usize>)>::default();
        for (index, (_, element_faces)) in elements.iter().enumerate() {
            for face in element_faces {
                let mut key = face.clone();
                key.sort_unstable();
                faces
                    .entry(key)
                    .or_insert_with(|| (face.clone(), Vec::new()))
                    .1
                    .push(index);
            }
        }
        if faces.values().any(|(_, owners)| owners.len() > 2) {
            return Err("a face is shared by more than two elements");
        }
        let mut parents: Vec<usize> = (0..elements.len()).collect();
        for (_, owners) in faces.values() {
            if let [first, second] = owners[..]
                && elements[first].0 == elements[second].0
            {
                let (first, second) = (root(&mut parents, first), root(&mut parents, second));
                parents[first] = second;
            }
        }
        let mut node_surfaces = vec![Vec::new(); self.number_of_nodes()];
        let mut edge_surfaces = FxHashMap::<[usize; 2], Vec<Surface>>::default();
        for (face, owners) in faces.values() {
            let volumes: Vec<usize> = owners.iter().map(|&o| root(&mut parents, o)).collect();
            let surface = match volumes[..] {
                [volume] => [volume, OUTSIDE],
                [first, second] if first != second => [first.min(second), first.max(second)],
                _ => continue,
            };
            face.iter().enumerate().for_each(|(i, &node)| {
                add(&mut node_surfaces[node], surface);
                let next = face[(i + 1) % face.len()];
                add(
                    edge_surfaces
                        .entry([node.min(next), node.max(next)])
                        .or_default(),
                    surface,
                );
            });
        }
        let mut curve_edges = vec![0; node_surfaces.len()];
        edge_surfaces
            .iter()
            .filter(|(_, surfaces)| surfaces.len() > 1)
            .for_each(|(edge, _)| edge.iter().for_each(|&node| curve_edges[node] += 1));
        Ok(node_surfaces
            .iter()
            .zip(curve_edges)
            .map(|(surfaces, curves)| match (surfaces.len(), curves) {
                (0, _) => NodeClass::Interior,
                (1, _) => NodeClass::Surface,
                (_, 2) => NodeClass::Curve,
                _ => NodeClass::Vertex,
            })
            .collect())
    }
}
