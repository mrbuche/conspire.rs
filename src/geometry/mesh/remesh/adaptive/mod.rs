#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Mesh, differential::jet::vertex_jets},
    },
    math::{Scalar, Tensor},
};

const D: usize = 3;
const N: usize = 3;

impl Mesh<3> {
    pub(super) fn adaptive_remesh(
        self,
        iterations: usize,
        tolerance: Scalar,
        minimum: Scalar,
        maximum: Scalar,
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
                    )
                },
            )?;
            Ok((vec![connectivity.into()], coordinates).into())
        }
    }
}

pub fn sizing_field(
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    tolerance: Scalar,
    minimum: Scalar,
    maximum: Scalar,
    gradation: Scalar,
) -> Vec<Scalar> {
    let mut field: Vec<Scalar> = vertex_jets(connectivity, coordinates)
        .into_iter()
        .map(|jet| {
            jet.map_or(maximum, |jet| {
                dunyach_length(jet.max_abs_curvature(), tolerance, minimum, maximum)
            })
        })
        .collect();
    graduate(&mut field, connectivity, coordinates, gradation);
    field
}

fn dunyach_length(
    curvature: Scalar,
    tolerance: Scalar,
    minimum: Scalar,
    maximum: Scalar,
) -> Scalar {
    if curvature <= 0.0 {
        return maximum;
    }
    let argument = 6.0 * tolerance / curvature - 3.0 * tolerance * tolerance;
    let length = if argument > 0.0 {
        argument.sqrt()
    } else {
        minimum
    };
    length.clamp(minimum, maximum)
}

fn graduate(
    field: &mut [Scalar],
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    gradation: Scalar,
) {
    let mut changed = true;
    while changed {
        changed = false;
        for &[a, b, c] in connectivity {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                let slope = gradation * (&coordinates[j] - &coordinates[i]).norm();
                if field[i] + slope < field[j] {
                    field[j] = field[i] + slope;
                    changed = true;
                } else if field[j] + slope < field[i] {
                    field[i] = field[j] + slope;
                    changed = true;
                }
            }
        }
    }
}
