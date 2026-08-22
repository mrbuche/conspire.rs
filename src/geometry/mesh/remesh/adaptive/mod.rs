use crate::{
    geometry::mesh::{
        Mesh,
        differential::sizing::{Creases, Unresolved, sizing_field},
    },
    math::{Quantity, Scalar},
    units::Length,
};

impl Mesh<3> {
    pub(crate) fn adaptive_remesh(
        self,
        iterations: usize,
        tolerance: Quantity<Length>,
        minimum: Quantity<Length>,
        maximum: Quantity<Length>,
        gradation: Scalar,
    ) -> Result<Self, &'static str> {
        if iterations == 0 {
            Ok(self)
        } else if self.connectivities().len() != 1 {
            Err("Can only remesh lone blocks for now.")
        } else {
            let (connectivities, mut coordinates) = self.into();
            let mut connectivity = Vec::try_from(connectivities)?;
            super::triangles::remesh(
                &mut connectivity,
                &mut coordinates,
                iterations,
                |connectivity, coordinates, _| {
                    sizing_field(
                        connectivity,
                        coordinates,
                        tolerance,
                        minimum,
                        maximum,
                        gradation,
                        Unresolved::Minimum,
                        Creases::Included,
                    )
                },
            )?;
            Ok((vec![connectivity.into()], coordinates).into())
        }
    }
}
