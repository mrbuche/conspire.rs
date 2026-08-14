#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Mesh, differential::jet::vertex_jets},
    },
    math::{Quantity, Scalar, Tensor},
    units::{Length, ReciprocalLength},
};

const D: usize = 3;
const N: usize = 3;

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
                    )
                },
            )?;
            Ok((vec![connectivity.into()], coordinates).into())
        }
    }
}

pub(crate) fn sizing_field(
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    tolerance: Quantity<Length>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
    gradation: Scalar,
) -> Vec<Quantity<Length>> {
    let mut field: Vec<Quantity<Length>> = vertex_jets(connectivity, coordinates)
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

/// The size a chord-error tolerance asks for at a given curvature.
///
/// Both terms under the root are areas, a tolerance over a curvature as much
/// as a tolerance squared, so the root is the length it gives back. Halving a
/// unit is not something the table names, so the root is taken in numbers and
/// a length asserted on the way out.
fn dunyach_length(
    curvature: Quantity<ReciprocalLength>,
    tolerance: Quantity<Length>,
    minimum: Quantity<Length>,
    maximum: Quantity<Length>,
) -> Quantity<Length> {
    if curvature <= Quantity::default() {
        return maximum;
    }
    let argument = tolerance * 6.0 / curvature - tolerance * tolerance * 3.0;
    let length = if argument > Quantity::default() {
        Quantity::new(argument.value().sqrt())
    } else {
        minimum
    };
    length.max(minimum).min(maximum)
}

fn graduate(
    field: &mut [Quantity<Length>],
    connectivity: &[[usize; N]],
    coordinates: &Coordinates<D>,
    gradation: Scalar,
) {
    let mut changed = true;
    while changed {
        changed = false;
        for &[a, b, c] in connectivity {
            for (i, j) in [(a, b), (b, c), (c, a)] {
                let slope = (&coordinates[j] - &coordinates[i]).norm() * gradation;
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
