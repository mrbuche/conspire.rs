#[cfg(test)]
mod test;

mod faces;
mod surface;

use crate::geometry::mesh::{Connectivity, Mesh};
use std::collections::HashMap;

pub use faces::ElementsFaces;
pub use surface::Surface;

/// The outward faces of each element, from which any set of elements has a boundary.
pub struct Boundary<S = Vec<Vec<Vec<usize>>>> {
    elements_faces: S,
}

impl TryFrom<&Mesh<3>> for Boundary {
    type Error = &'static str;
    fn try_from(mesh: &Mesh<3>) -> Result<Self, &'static str> {
        let mut elements_faces = Vec::with_capacity(mesh.number_of_elements());
        for block in mesh.iter() {
            match block {
                Connectivity::Polyhedral(connectivity) => {
                    let mut owner = HashMap::new();
                    connectivity.elements_faces().iter().enumerate().for_each(
                        |(element, faces)| {
                            faces.iter().for_each(|&face| {
                                owner.entry(face).or_insert(element);
                            })
                        },
                    );
                    connectivity
                        .elements_faces()
                        .iter()
                        .enumerate()
                        .for_each(|(element, faces)| {
                            elements_faces.push(
                                faces
                                    .iter()
                                    .map(|&face| {
                                        let mut nodes = connectivity.faces_nodes()[face].clone();
                                        if owner[&face] != element {
                                            nodes.reverse()
                                        }
                                        nodes
                                    })
                                    .collect(),
                            )
                        })
                }
                Connectivity::Polygonal(_)
                | Connectivity::Quadrilateral(_)
                | Connectivity::Triangular(_) => {
                    return Err("a boundary requires three-dimensional elements");
                }
                _ => block
                    .iter()
                    .for_each(|element| elements_faces.push(block.element_faces(element))),
            }
        }
        Ok(Self { elements_faces })
    }
}

impl<S: ElementsFaces> Boundary<S> {
    pub fn new(elements_faces: S) -> Self {
        Self { elements_faces }
    }
    pub fn number_of_elements(&self) -> usize {
        self.elements_faces.number_of_elements()
    }
    /// The elements that share a face with each element.
    pub fn adjacent(&self) -> Vec<Vec<usize>> {
        let mut faces = HashMap::<Vec<usize>, Vec<usize>>::new();
        (0..self.number_of_elements()).for_each(|element| {
            self.elements_faces.element_faces(element).for_each(|face| {
                let mut key = face.as_ref().to_vec();
                key.sort_unstable();
                faces.entry(key).or_default().push(element)
            })
        });
        let mut adjacent = vec![Vec::new(); self.number_of_elements()];
        faces.values().for_each(|elements| {
            elements.iter().for_each(|&element| {
                adjacent[element].extend(elements.iter().filter(|&&other| other != element))
            })
        });
        adjacent
    }
    /// The faces on the boundary of the union of the elements, connected through shared faces.
    pub fn faces(&self, elements: &[usize]) -> Result<Vec<Vec<usize>>, &'static str> {
        if elements.is_empty() {
            return Err("there are no elements");
        }
        let mut roots = (0..elements.len()).collect::<Vec<_>>();
        let mut tally = HashMap::<Vec<usize>, (usize, usize)>::new();
        let mut order = Vec::<(Vec<usize>, Vec<usize>)>::new();
        for (position, &element) in elements.iter().enumerate() {
            for face in self.elements_faces.element_faces(element) {
                let face = face.as_ref();
                let mut key = face.to_vec();
                key.sort_unstable();
                match tally.get_mut(&key) {
                    Some((count, first)) => {
                        *count += 1;
                        let (a, b) = (find(&mut roots, *first), find(&mut roots, position));
                        roots[b] = a;
                    }
                    None => {
                        tally.insert(key.clone(), (1, position));
                        order.push((face.to_vec(), key))
                    }
                }
            }
        }
        let root = find(&mut roots, 0);
        if (1..elements.len()).any(|position| find(&mut roots, position) != root) {
            return Err("the elements are not connected through shared faces");
        }
        Ok(order
            .into_iter()
            .filter(|(_, key)| tally[key].0 == 1)
            .map(|(face, _)| face)
            .collect())
    }
    /// The topology of the boundary of the union of the elements.
    pub fn surface(&self, elements: &[usize]) -> Result<Surface, String> {
        self.faces(elements)?.as_slice().try_into()
    }
}

fn find(roots: &mut [usize], mut node: usize) -> usize {
    while roots[node] != node {
        roots[node] = roots[roots[node]];
        node = roots[node]
    }
    node
}
