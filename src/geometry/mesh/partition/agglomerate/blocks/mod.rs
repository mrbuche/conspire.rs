#[cfg(test)]
mod test;

use super::super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Boundary, Connectivities, Connectivity, Mesh, PolytopalConnectivity,
            PrimitiveConnectivity,
        },
    },
    math::{Set, Tensor, TensorVec},
};
use std::collections::HashMap;

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
        let mut remap = vec![usize::MAX; mesh.number_of_nodes()];
        blocks.iter().for_each(|block| {
            block.iter().for_each(|element| {
                block
                    .element_nodes(element)
                    .into_iter()
                    .for_each(|node| remap[node] = 0)
            })
        });
        let mut coordinates = Coordinates::new();
        remap.iter_mut().enumerate().for_each(|(node, new)| {
            if *new == 0 {
                *new = coordinates.len();
                coordinates.push(mesh.coordinates()[node].clone())
            }
        });
        let blocks = blocks
            .iter()
            .map(|block| renumbered(block, &remap))
            .collect::<Vec<_>>();
        let connectivities = if mesh.blocks().is_some() {
            Connectivities::from((blocks, numbers))
        } else {
            Connectivities::from(blocks)
        };
        Ok((connectivities, Set::from(coordinates)).into())
    }
}

fn polyhedra(boundary: &Boundary, parts: &[&[usize]]) -> Result<Connectivity, &'static str> {
    let mut faces_nodes = Vec::<Vec<usize>>::new();
    let mut indices = HashMap::<Vec<usize>, usize>::new();
    let mut cells = Vec::new();
    for elements in parts {
        cells.push(
            boundary
                .faces(elements)?
                .into_iter()
                .map(|face| {
                    let mut key = face.clone();
                    key.sort_unstable();
                    *indices.entry(key).or_insert_with(|| {
                        faces_nodes.push(face);
                        faces_nodes.len() - 1
                    })
                })
                .collect::<Vec<usize>>(),
        )
    }
    Ok(Connectivity::Polyhedral((cells, faces_nodes).into()))
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

fn renumbered(block: &Connectivity, remap: &[usize]) -> Connectivity {
    fn map<const N: usize>(
        connectivity: &PrimitiveConnectivity<3, N>,
        remap: &[usize],
    ) -> PrimitiveConnectivity<3, N> {
        connectivity
            .iter()
            .map(|nodes| nodes.map(|node| remap[node]))
            .collect::<Vec<_>>()
            .into()
    }
    match block {
        Connectivity::Hexahedral(c) => Connectivity::Hexahedral(map(c, remap)),
        Connectivity::Pyramidal(c) => Connectivity::Pyramidal(map(c, remap)),
        Connectivity::Tetrahedral(c) => Connectivity::Tetrahedral(map(c, remap)),
        Connectivity::Wedge(c) => Connectivity::Wedge(map(c, remap)),
        Connectivity::Polyhedral(c) => Connectivity::Polyhedral(PolytopalConnectivity::from((
            c.elements_faces().to_vec(),
            c.faces_nodes()
                .iter()
                .map(|face| face.iter().map(|&node| remap[node]).collect())
                .collect(),
        ))),
        _ => unreachable!("a boundary requires three-dimensional elements"),
    }
}
