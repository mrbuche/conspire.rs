use crate::{
    domain::{
        Blocks, ElementModelError, Model, NodalCoordinates, NodalReferenceCoordinates,
        block::element::Elements,
    },
    math::{Quantity, Scalar},
    units::Time,
};

/// Elements that report how fast their motion can be.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub trait TimeScaleElements<const D: usize>
where
    Self: Elements,
{
    /// The shortest time scale among the elements at the given coordinates, the
    /// reciprocal of the highest angular frequency, which bounds a stable explicit time step.
    fn fastest_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError>;
}

impl<B, const D: usize> TimeScaleElements<D> for Model<B, D>
where
    B: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        self.blocks
            .fastest_time_scale(reference_coordinates, nodal_coordinates)
    }
}

impl<B1, B2, const D: usize> TimeScaleElements<D> for Blocks<B1, B2>
where
    B1: TimeScaleElements<D>,
    B2: TimeScaleElements<D>,
{
    fn fastest_time_scale(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_coordinates: &NodalCoordinates<D>,
    ) -> Result<Quantity<Time>, ElementModelError> {
        Ok(self
            .0
            .fastest_time_scale(reference_coordinates, nodal_coordinates)?
            .min(
                self.1
                    .fastest_time_scale(reference_coordinates, nodal_coordinates)?,
            ))
    }
}

const MAXIMUM_ITERATIONS: usize = 500;
const TOLERANCE: Scalar = 1e-10;

/// The largest eigenvalue of $`M^{-1}K`$, by power iteration, for the stiffness
/// $`K`$ of `size` degrees of freedom and the lumped masses $`M`$ on them.
///
/// Iterates from the same deterministic, non-symmetric start every time, until
/// the Rayleigh quotient changes by less than a relative tolerance. The
/// estimate approaches the largest eigenvalue from below, so it can fall short
/// of it by about the tolerance, or by more where the largest eigenvalues are
/// clustered and the iteration stops early.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn largest_eigenvalue(
    size: usize,
    stiffness: impl Fn(usize, usize) -> Scalar,
    masses: &[Scalar],
) -> Scalar {
    assert_eq!(
        size,
        masses.len(),
        "There must be a mass for each degree of freedom."
    );
    let mut vector: Vec<Scalar> = (0..size)
        .map(|index| ((index as u64 * 2654435761) % 4294967296) as Scalar / 4294967296.0 - 0.5)
        .collect();
    let mut eigenvalue = 0.0;
    for _ in 0..MAXIMUM_ITERATIONS {
        let product: Vec<Scalar> = (0..size)
            .map(|row| {
                (0..size)
                    .map(|column| stiffness(row, column) * vector[column])
                    .sum()
            })
            .collect();
        let energy: Scalar = vector.iter().zip(&product).map(|(v, p)| v * p).sum();
        let mass: Scalar = vector.iter().zip(masses).map(|(v, m)| v * v * m).sum();
        let previous = eigenvalue;
        eigenvalue = energy / mass;
        let next: Vec<Scalar> = product.iter().zip(masses).map(|(p, m)| p / m).collect();
        let norm = next
            .iter()
            .zip(masses)
            .map(|(w, m)| w * w * m)
            .sum::<Scalar>()
            .sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        vector = next.into_iter().map(|w| w / norm).collect();
        if (eigenvalue - previous).abs() <= TOLERANCE * eigenvalue.abs() {
            break;
        }
    }
    eigenvalue
}

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn time_scale_from_eigenvalue(eigenvalue: Scalar) -> Quantity<Time> {
    if eigenvalue > 0.0 {
        Time::seconds(1.0 / eigenvalue.sqrt())
    } else {
        Time::seconds(Scalar::INFINITY)
    }
}
