#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Mesh, NodeSets, SideSets},
    },
    math::Tensor,
};

fn accumulate_numbers(accumulator: &mut Option<Vec<usize>>, next: Option<&[usize]>) {
    match (accumulator.as_mut(), next) {
        (Some(numbers), Some(next)) => numbers.extend_from_slice(next),
        _ => *accumulator = None,
    }
}

/// Fuses independently-built meshes into one, with no regard for whether
/// they are geometrically conforming at any shared boundary.
///
/// Every node index a later mesh's blocks, node sets, and side sets refer to
/// is shifted past those of every mesh before it, so each input keeps its
/// own connectivity untouched other than that shift. Block, node set, and
/// side set numbers are carried over only when every input mesh provides
/// them; otherwise the combined mesh leaves that numbering unset rather than
/// mixing numbered and unnumbered blocks/sets.
impl<const D: usize> FromIterator<Mesh<D>> for Mesh<D> {
    fn from_iter<T: IntoIterator<Item = Mesh<D>>>(iter: T) -> Self {
        let mut connectivities = Vec::new();
        let mut connectivity_numbers = Some(Vec::new());
        let mut coordinates = Coordinates::<D>::from(Vec::<[f64; D]>::new());
        let mut node_sets = Vec::new();
        let mut node_set_numbers = Some(Vec::new());
        let mut side_sets = Vec::new();
        let mut side_set_numbers = Some(Vec::new());
        let mut node_offset = 0;
        let mut element_offset = 0;
        iter.into_iter().for_each(|mesh| {
            let number_of_nodes = mesh.number_of_nodes();
            let number_of_elements = mesh.number_of_elements();
            connectivities.extend(mesh.iter().map(|block| block.offset(node_offset)));
            accumulate_numbers(&mut connectivity_numbers, mesh.blocks());
            coordinates.extend(mesh.coordinates().iter().cloned());
            node_sets.extend(mesh.node_sets().iter().map(|set| {
                set.iter()
                    .map(|&node| node + node_offset)
                    .collect::<Vec<_>>()
            }));
            accumulate_numbers(&mut node_set_numbers, mesh.node_set_numbers());
            side_sets.extend(mesh.side_sets().iter().map(|set| {
                set.iter()
                    .map(|&(element, side)| (element + element_offset, side))
                    .collect::<Vec<_>>()
            }));
            accumulate_numbers(&mut side_set_numbers, mesh.side_set_numbers());
            node_offset += number_of_nodes;
            element_offset += number_of_elements;
        });
        let connectivities = match connectivity_numbers {
            Some(numbers) => Connectivities::from((connectivities, numbers)),
            None => Connectivities::from(connectivities),
        };
        let mut mesh = Mesh::from((connectivities, coordinates.into()));
        mesh.set_node_sets(match node_set_numbers {
            Some(numbers) => NodeSets::from((node_sets, numbers)),
            None => NodeSets::from(node_sets),
        });
        mesh.set_side_sets(match side_set_numbers {
            Some(numbers) => SideSets::from((side_sets, numbers)),
            None => SideSets::from(side_sets),
        });
        mesh
    }
}
