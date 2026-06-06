use crate::geometry::{Coordinates, mesh::Mesh};

impl<const D: usize> Mesh<D> {
    pub fn isotropic_remesh(self, iterations: usize) -> Result<(), &'static str> {
        if iterations == 0 {
            Ok(())
        } else if self.connectivities().len() == 1 {
            Err("Can only remesh single blocks for now.")
        } else {
            let (connectivities, coordinates) = self.into();
            let connectivity = Vec::try_from(connectivities)?;
            foo(connectivity, coordinates)
        }
    }
}

fn foo<const D: usize>(
    connectivity: Vec<[usize; 3]>,
    coordinates: Coordinates<D>,
) -> Result<(), &'static str> {
    Ok(())
}
