use super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivities, Mesh, NodeSets, SideSets, retain::subset},
    },
    math::{Set, TensorVec},
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
        let mut block_numbers = Vec::new();
        for (block, connectivity) in mesh.iter().enumerate() {
            let mut kept = Vec::new();
            let mut element_numbers = Vec::new();
            for (local, element) in connectivity.iter().enumerate() {
                let global = index + local;
                if elements[global] != usize::MAX {
                    kept.push(element);
                    element_numbers.push(match connectivity.element_numbers() {
                        Some(numbers) => numbers[local],
                        None => global + 1,
                    })
                }
            }
            index += connectivity.number_of_elements();
            if !kept.is_empty() {
                let mut new_block = subset(connectivity, &kept, &mut |node| nodes[node]);
                new_block.number_elements(element_numbers);
                blocks.push(new_block);
                block_numbers.push(mesh.blocks().map_or(block + 1, |numbers| numbers[block]))
            }
        }
        let mut coordinates = Coordinates::new();
        old_nodes
            .iter()
            .for_each(|&node| coordinates.push(mesh.coordinates()[node].clone()));
        let node_numbers = old_nodes
            .iter()
            .map(|&node| {
                mesh.coordinates
                    .numbers()
                    .map_or(node + 1, |numbers| numbers[node])
            })
            .collect::<Vec<_>>();
        let mut submesh: Mesh<D> = (
            Connectivities::from((blocks, block_numbers)),
            Set::from((coordinates, node_numbers)),
        )
            .into();
        let (mut node_sets, mut node_set_numbers) = (Vec::new(), Vec::new());
        mesh.node_sets()
            .iter()
            .enumerate()
            .for_each(|(set, members)| {
                let kept: Vec<usize> = members
                    .iter()
                    .filter(|&&node| nodes[node] != usize::MAX)
                    .map(|&node| nodes[node])
                    .collect();
                if !kept.is_empty() {
                    node_sets.push(kept);
                    node_set_numbers.push(
                        mesh.node_set_numbers()
                            .map_or(set + 1, |numbers| numbers[set]),
                    )
                }
            });
        let (mut side_sets, mut side_set_numbers) = (Vec::new(), Vec::new());
        mesh.side_sets()
            .iter()
            .enumerate()
            .for_each(|(set, members)| {
                let kept: Vec<(usize, usize)> = members
                    .iter()
                    .filter(|&&(element, _)| elements[element] != usize::MAX)
                    .map(|&(element, side)| (elements[element], side))
                    .collect();
                if !kept.is_empty() {
                    side_sets.push(kept);
                    side_set_numbers.push(
                        mesh.side_set_numbers()
                            .map_or(set + 1, |numbers| numbers[set]),
                    )
                }
            });
        submesh.set_node_sets(NodeSets::from((node_sets, node_set_numbers)));
        submesh.set_side_sets(SideSets::from((side_sets, side_set_numbers)));
        (submesh, old_nodes)
    }
}
