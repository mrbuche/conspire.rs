use crate::{
    geometry::mesh::Mesh,
    math::{Quantity, Scalar, Tensor, unit::Length},
};

impl<const D: usize> Mesh<D> {
    pub(crate) fn uniform_remesh(
        self,
        iterations: usize,
        length: Option<Quantity<Length>>,
    ) -> Result<Self, &'static str> {
        if iterations == 0 {
            Ok(self)
        } else if self.connectivities().len() != 1 {
            Err("Can only remesh lone blocks for now.")
        } else {
            let (connectivities, mut coordinates) = self.into();
            let mut connectivity = Vec::try_from(connectivities)?;
            let mut target = length;
            super::triangles::remesh(
                &mut connectivity,
                &mut coordinates,
                iterations,
                |_, coordinates, lengths| {
                    let target = *target.get_or_insert_with(|| {
                        lengths.values().copied().sum::<Quantity<Length>>()
                            / lengths.len() as Scalar
                    });
                    vec![target; coordinates.len()]
                },
            )?;
            Ok((vec![connectivity.into()], coordinates).into())
        }
    }
}
