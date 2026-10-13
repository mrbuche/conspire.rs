#[cfg(test)]
mod test;

use super::{super::Partition, compact, polyhedra};
use crate::{
    geometry::mesh::{Boundary, Connectivities, Connectivity, Mesh, PrimitiveConnectivity},
    math::Set,
};

impl Partition {
    /// Agglomerates each part of more than one element into a polyhedron, leaving the others.
    ///
    /// Every block keeps its elements that are alone in their part, in a block of the same
    /// type. The polyhedra of a block go in one polyhedral block right after it, numbered above
    /// all of the blocks when the mesh has block numbers. The elements of a polyhedral block
    /// stay polyhedra in that one block. Nodes that no element uses any longer are removed.
    pub fn agglomerate_blocks(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        let boundary = Boundary::try_from(mesh)?;
        let elements_blocks = mesh.elements_blocks();
        if (0..self.number_of_parts()).any(|part| {
            self.part_elements(part)
                .windows(2)
                .any(|pair| elements_blocks[pair[0]] != elements_blocks[pair[1]])
        }) {
            return Err("a part has elements in several blocks");
        }
        let alone = |element: usize| self.part_elements(self.part_of(element)).len() == 1;
        let mut next = mesh
            .blocks()
            .map_or(0, |numbers| numbers.iter().max().map_or(0, |max| max + 1));
        let mut blocks = Vec::new();
        let mut numbers = Vec::new();
        let mut offset = 0;
        for (index, block) in mesh.iter().enumerate() {
            let number = mesh.blocks().map(|numbers| numbers[index]);
            let parts = (0..self.number_of_parts())
                .filter(|&part| {
                    self.part_elements(part)
                        .first()
                        .is_some_and(|&element| elements_blocks[element] == index)
                        && (matches!(block, Connectivity::Polyhedral(_))
                            || self.part_elements(part).len() > 1)
                })
                .map(|part| self.part_elements(part))
                .collect::<Vec<_>>();
            if let Connectivity::Polyhedral(_) = block {
                blocks.push(polyhedra(&boundary, &parts)?);
                numbers.extend(number);
            } else {
                let kept = kept(block, offset, &alone);
                if !kept.is_empty() {
                    blocks.push(kept);
                    numbers.extend(number);
                }
                if !parts.is_empty() {
                    blocks.push(polyhedra(&boundary, &parts)?);
                    if number.is_some() {
                        numbers.push(next);
                        next += 1
                    }
                }
            }
            offset += block.number_of_elements();
        }
        let (blocks, coordinates) = compact(mesh, blocks);
        let connectivities = if mesh.blocks().is_some() {
            Connectivities::from((blocks, numbers))
        } else {
            Connectivities::from(blocks)
        };
        Ok((connectivities, Set::from(coordinates)).into())
    }
}

fn kept(block: &Connectivity, offset: usize, keep: &impl Fn(usize) -> bool) -> Connectivity {
    fn select<const N: usize>(
        connectivity: &PrimitiveConnectivity<3, N>,
        offset: usize,
        keep: &impl Fn(usize) -> bool,
    ) -> PrimitiveConnectivity<3, N> {
        connectivity
            .iter()
            .enumerate()
            .filter(|(element, _)| keep(offset + element))
            .map(|(_, nodes)| *nodes)
            .collect::<Vec<_>>()
            .into()
    }
    match block {
        Connectivity::Hexahedral(c) => Connectivity::Hexahedral(select(c, offset, keep)),
        Connectivity::Pyramidal(c) => Connectivity::Pyramidal(select(c, offset, keep)),
        Connectivity::Tetrahedral(c) => Connectivity::Tetrahedral(select(c, offset, keep)),
        Connectivity::Wedge(c) => Connectivity::Wedge(select(c, offset, keep)),
        _ => unreachable!("a boundary requires three-dimensional elements"),
    }
}
