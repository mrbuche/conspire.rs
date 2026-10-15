use super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Mesh, NodeSet, SideSet, retain::subset},
    },
    math::TensorVec,
};

impl Partition {
    pub fn part<const D: usize>(&self, mesh: &Mesh<D>, part: usize) -> (Mesh<D>, Vec<usize>) {
        let old_nodes = self.parts_nodes[part].clone();
        let mut nodes = vec![usize::MAX; mesh.number_of_nodes()];
        old_nodes
            .iter()
            .enumerate()
            .for_each(|(new, &old)| nodes[old] = new);
        let mut elements = vec![usize::MAX; mesh.number_of_elements()];
        self.parts_elements[part]
            .iter()
            .enumerate()
            .for_each(|(new, &old)| elements[old] = new);
        let mut index = 0;
        let mut blocks = Vec::new();
        let mut block_labels = Vec::new();
        for (block, connectivity) in mesh.iter().enumerate() {
            let mut kept = Vec::new();
            let mut element_labels = Vec::new();
            for (local, element) in connectivity.iter().enumerate() {
                let global = index + local;
                if elements[global] != usize::MAX {
                    kept.push(element);
                    element_labels.push(match connectivity.element_labels() {
                        Some(labels) => labels[local],
                        None => global + 1,
                    })
                }
            }
            index += connectivity.number_of_elements();
            if !kept.is_empty() {
                let mut new_block = subset(connectivity, &kept, &mut |node| nodes[node]);
                new_block.label_elements(element_labels);
                blocks.push(new_block);
                block_labels.push(
                    mesh.block_labels()
                        .map_or(block + 1, |labels| labels[block]),
                )
            }
        }
        let mut coordinates = Coordinates::new();
        old_nodes
            .iter()
            .for_each(|&node| coordinates.push(mesh.coordinates()[node].clone()));
        let node_labels = old_nodes
            .iter()
            .map(|&node| mesh.node_labels().map_or(node + 1, |labels| labels[node]))
            .collect::<Vec<_>>();
        let mut submesh: Mesh<D> = (
            Connectivities::from((blocks, block_labels)),
            coordinates,
            node_labels,
        )
            .into();
        let node_sets: Vec<NodeSet> = mesh
            .node_sets()
            .iter()
            .enumerate()
            .filter_map(|(set, node_set)| {
                node_set
                    .restrict(|node| (nodes[node] != usize::MAX).then_some(nodes[node]))
                    .map(|kept| match kept.label() {
                        Some(_) => kept,
                        None => kept.with_label(set + 1),
                    })
            })
            .collect();
        let side_sets: Vec<SideSet> = mesh
            .side_sets()
            .iter()
            .enumerate()
            .filter_map(|(set, side_set)| {
                side_set
                    .restrict(|(element, side)| {
                        (elements[element] != usize::MAX).then_some((elements[element], side))
                    })
                    .map(|kept| match kept.label() {
                        Some(_) => kept,
                        None => kept.with_label(set + 1),
                    })
            })
            .collect();
        submesh.set_node_sets(node_sets.into());
        submesh.set_side_sets(side_sets.into());
        (submesh, old_nodes)
    }
}
