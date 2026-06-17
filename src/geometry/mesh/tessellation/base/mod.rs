#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            smooth::Smoothing,
            tessellation::{D, Normals, Tessellation},
        },
    },
    math::{Tensor, TensorVec},
};
use std::{cell::OnceCell, collections::HashMap};

impl Tessellation {
    pub fn mesh(&self) -> &Mesh<D> {
        &self.mesh
    }
    pub fn normals(&self) -> &Normals {
        &self.normals
    }
    pub fn bvh(&self) -> &BoundingVolumeHierarchy<D> {
        self.bvh
            .get_or_init(|| BoundingVolumeHierarchy::from(&self.mesh))
    }
    pub fn smooth(&mut self, smoothing: Smoothing) {
        self.mesh.smooth(smoothing);
        self.refresh();
    }
    pub fn smooth_welded(&mut self, smoothing: Smoothing) {
        let mut representatives = Vec::with_capacity(self.mesh.number_of_nodes());
        let mut groups = HashMap::new();
        let mut welded = Coordinates::new();
        for point in self.mesh.coordinates() {
            let key = [point[0].to_bits(), point[1].to_bits(), point[2].to_bits()];
            representatives.push(*groups.entry(key).or_insert_with(|| {
                let representative = welded.len();
                welded.push(point.clone());
                representative
            }));
        }
        let triangles = match &self.mesh.connectivities()[0] {
            Connectivity::Triangular(triangles) => triangles
                .iter()
                .map(|&[a, b, c]| [representatives[a], representatives[b], representatives[c]])
                .collect::<Vec<_>>(),
            _ => panic!(),
        };
        let mut mesh = Mesh::from((vec![Connectivity::Triangular(triangles.into())], welded));
        mesh.smooth(smoothing);
        let smoothed = mesh.coordinates();
        self.mesh
            .coordinates
            .iter_mut()
            .zip(&representatives)
            .for_each(|(point, &representative)| *point = smoothed[representative].clone());
        self.refresh();
    }
    fn refresh(&mut self) {
        self.normals = self.mesh.normals();
        self.bvh = OnceCell::new();
    }
}
