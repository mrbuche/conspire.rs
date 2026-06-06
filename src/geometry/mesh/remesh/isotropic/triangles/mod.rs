use crate::geometry::Coordinates;

pub fn isotropic_remesh<const D: usize>(
    connectivity: &mut Vec<[usize; 3]>,
    coordinates: &mut Coordinates<D>,
) -> Result<(), &'static str> {
    Ok(())
}
