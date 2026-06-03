use crate::{
    geometry::{
        mesh::{
            Mesh,
            tessellation::{D, Tessellation},
        },
        ntree::{Balance, Balancing, Dualization, Octree, OrthotreeError, Pairing},
    },
    math::Scalar,
};

impl Tessellation {
    pub fn dualize(&self, scale: Scalar) -> Result<Mesh<D>, OrthotreeError> {
        let mut octree = Octree::from_sdf(self, scale);
        octree.equilibrate(Balancing::Strong, Pairing::Regular)?;
        let mut mesh = octree.dualize();
        self.trim(&mut mesh);
        Ok(mesh)
    }
    fn trim(&self, mesh: &mut Mesh<D>) {
        // need to trim nodes outside tessellation
    }
}
