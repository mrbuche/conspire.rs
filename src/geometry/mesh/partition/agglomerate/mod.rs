#[cfg(test)]
mod test;

mod blocks;

use super::Partition;
use crate::{
    geometry::{
        Coordinates,
        mesh::{
            Boundary, Connectivities, Connectivity, Mesh, PolytopalConnectivity,
            PrimitiveConnectivity,
        },
    },
    math::{Tensor, TensorVec},
};
use std::collections::HashMap;

impl Partition {
    pub fn agglomerate(&self, mesh: &Mesh<3>) -> Result<Mesh<3>, &'static str> {
        let boundary = Boundary::try_from(mesh)?;
        let parts = (0..self.number_of_parts())
            .map(|part| self.part_elements(part))
            .filter(|elements| !elements.is_empty())
            .collect::<Vec<_>>();
        let (blocks, coordinates) = compact(mesh, vec![polyhedra(&boundary, &parts)?]);
        Ok((Connectivities::from(blocks), coordinates).into())
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

fn compact(mesh: &Mesh<3>, blocks: Vec<Connectivity>) -> (Vec<Connectivity>, Coordinates<3>) {
    let mut remap = vec![usize::MAX; mesh.number_of_nodes()];
    blocks.iter().for_each(|block| match block {
        Connectivity::Polyhedral(connectivity) => connectivity
            .faces_nodes()
            .iter()
            .flatten()
            .for_each(|&node| remap[node] = 0),
        _ => block.iter().for_each(|element| {
            block
                .element_nodes(element)
                .into_iter()
                .for_each(|node| remap[node] = 0)
        }),
    });
    let mut coordinates = Coordinates::new();
    remap.iter_mut().enumerate().for_each(|(node, new)| {
        if *new == 0 {
            *new = coordinates.len();
            coordinates.push(mesh.coordinates()[node].clone())
        }
    });
    (
        blocks
            .iter()
            .map(|block| renumbered(block, &remap))
            .collect(),
        coordinates,
    )
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
