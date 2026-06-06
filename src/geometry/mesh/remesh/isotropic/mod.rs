mod triangles;

use crate::geometry::mesh::Mesh;

impl<const D: usize> Mesh<D> {
    pub fn isotropic_remesh(self, iterations: usize) -> Result<Self, &'static str> {
        if iterations == 0 {
            Ok(self)
        } else if self.connectivities().len() == 1 {
            Err("Can only remesh lone blocks for now.")
        } else {
            let (connectivities, mut coordinates) = self.into();
            let mut connectivity = Vec::try_from(connectivities)?;
            triangles::isotropic_remesh(&mut connectivity, &mut coordinates)?;
            Ok((vec![connectivity.into()], coordinates).into())
        }
    }
}
