use crate::{
    geometry::{
        Coordinate, Coordinates,
        bvh::BoundingVolumeHierarchy,
        mesh::{
            Connectivity, Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, Dualization, Octree, OrthotreeError, Pairing},
    },
    math::{Scalar, Tensor, TensorVec},
};
use std::array::from_fn;

const N: usize = 8;

impl Tessellation {
    pub fn dualize(&self, scale: Scalar) -> Result<Mesh<D>, OrthotreeError> {
        let (mut octree, bvh) = Octree::from_sdf(self, scale);
        octree.equilibrate(Balancing::Strong, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh, &bvh);
        Ok(mesh)
    }
    /// Drops every dual element with a node outside the tessellation, reusing the
    /// surface BVH from the shape diameter function for the inside/outside tests.
    fn trim(&self, mesh: &mut Mesh<D>, bvh: &BoundingVolumeHierarchy<D>) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        // A skewed direction avoids grazing the axis-aligned dual faces and
        // surface edges, which would corrupt the crossing parity.
        let direction: Coordinate<D> = [1.0, 0.140_412_03, 0.092_153_88].into();
        let inside: Vec<bool> = mesh
            .coordinates()
            .iter()
            .map(|point| {
                let ray = (point.clone(), direction.clone()).into();
                bvh.intersections(&ray, surface_coordinates, &elements) % 2 == 1
            })
            .collect();
        let mut remap = vec![usize::MAX; inside.len()];
        let mut coordinates = Coordinates::new();
        let mut connectivity: Vec<[usize; N]> = Vec::new();
        mesh.iter()
            .flatten()
            .filter(|element| element.iter().all(|&node| inside[node]))
            .for_each(|element| {
                connectivity.push(from_fn(|i| {
                    let node = element[i];
                    if remap[node] == usize::MAX {
                        remap[node] = coordinates.len();
                        coordinates.push(mesh.coordinates()[node].clone());
                    }
                    remap[node]
                }))
            });
        *mesh = (
            vec![Connectivity::Hexahedral(connectivity.into())],
            coordinates,
        )
            .into();
    }
}
