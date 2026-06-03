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

const GRAZING_TOLERANCE: Scalar = 1.0e-4;
const DIRECTIONS: [Coordinate<D>; 3] = [
    Coordinate::const_from([1.0, 0.140_412_03, 0.092_153_88]),
    Coordinate::const_from([0.097_153_2, 1.0, 0.131_771_4]),
    Coordinate::const_from([0.123_456_7, 0.087_654_3, 1.0]),
];

impl Tessellation {
    pub fn dualize(&self, scale: Scalar) -> Result<Mesh<D>, OrthotreeError> {
        let (mut octree, bvh) = Octree::from_sdf(self, scale);
        octree.equilibrate(Balancing::Strong, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh, &bvh);
        Ok(mesh)
    }
    fn trim(&self, mesh: &mut Mesh<D>, bvh: &BoundingVolumeHierarchy<D>) {
        let surface = self.mesh();
        let surface_coordinates = surface.coordinates();
        let elements: Vec<&[usize]> = surface.connectivities().iter().flatten().collect();
        let normals: Vec<&Coordinate<D>> = self.normals().iter().flatten().collect();
        let directions = DIRECTIONS.map(|direction| direction.normalized());
        let inside: Vec<bool> = mesh
            .coordinates()
            .iter()
            .map(|point| {
                directions
                    .iter()
                    .find_map(|direction| {
                        let ray = (point.clone(), direction.clone()).into();
                        match bvh.intersect(&ray, surface_coordinates, &elements) {
                            None => Some(false),
                            Some(hit) => {
                                let normal = normals[hit.index()];
                                let cosine = (direction * normal) / normal.norm();
                                (cosine.abs() > GRAZING_TOLERANCE).then_some(cosine > 0.0)
                            }
                        }
                    })
                    .unwrap_or(false)
            })
            .collect();
        let mut remap = vec![usize::MAX; inside.len()];
        let mut coordinates = Coordinates::new();
        let mut connectivity = Vec::new();
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
