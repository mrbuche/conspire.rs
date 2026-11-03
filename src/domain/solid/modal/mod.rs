#[cfg(test)]
pub(crate) mod test;

use crate::{
    domain::{
        Model, NodalCoordinates,
        factor::FreeFactored,
        solid::{
            NodalStiffnessesSolid, elastic::ElasticElements,
            elastic_viscoplastic::ElasticViscoplasticElements, mass::InertialMass,
        },
    },
    math::{
        Current, Differentiable, Quantity, Scalar, Style, StyledError, Tensor, TensorRank1Vec,
        Vector, assert::AssertionError, optimize::Lanczos, styled_error,
    },
    units::Rate,
};
use std::fmt::Display;

/// The natural frequencies of a solid and the shapes it vibrates in.
#[derive(Clone, Debug)]
pub struct Modes<const D: usize> {
    /// Angular frequencies, ascending.
    pub frequencies: Vec<Quantity<Rate>>,
    /// Shapes with zero entries at the fixed degrees of freedom, normalized
    /// so that the mass they carry is one in SI units.
    pub shapes: Vec<TensorRank1Vec<D, Current>>,
}

/// Vibration of elastic solids about a given state.
pub trait ElasticModes<const D: usize> {
    /// The `count` modes with angular frequencies nearest `near`, of small
    /// vibrations about the given coordinates.
    ///
    /// The stiffness is the tangent at the coordinates, which is the elastic
    /// vibration of a body at rest, or about a solved loaded state. Degrees of
    /// freedom are indexed as `D * node + component`. Without fixed degrees of
    /// freedom a body keeps its rigid-body modes at zero frequency.
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<Modes<D>, ModalError>;
}

impl<B, const D: usize> ElasticModes<D> for Model<B, D>
where
    B: ElasticElements<D>,
{
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<Modes<D>, ModalError> {
        let tangent = self
            .nodal_stiffnesses(nodal_coordinates)
            .map_err(|error| ModalError::upstream(error, "the stiffnesses"))?;
        modes_of(tangent, nodal_coordinates.len(), masses, fixed, near, count)
    }
}

/// Vibration of elastic-viscoplastic solids about a given state.
pub trait ElasticViscoplasticModes<S, const D: usize> {
    /// The `count` modes with angular frequencies nearest `near`, of small
    /// vibrations about the given coordinates and state variables.
    ///
    /// The stiffness is the elastic tangent with the plastic deformation gradient
    /// held at the given state: the instantaneous response, with no flow during the
    /// vibration, which is exact when the vibration is much faster than the flow.
    /// Neither the damping nor the relaxation of the flow is accounted for.
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<Modes<D>, ModalError>;
}

impl<B, S, const D: usize> ElasticViscoplasticModes<S, D> for Model<B, D>
where
    B: ElasticViscoplasticElements<S, D>,
    S: Differentiable,
{
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        state_variables: &S,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<Modes<D>, ModalError> {
        let tangent = self
            .nodal_stiffnesses(nodal_coordinates, state_variables)
            .map_err(|error| ModalError::upstream(error, "the stiffnesses"))?;
        modes_of(tangent, nodal_coordinates.len(), masses, fixed, near, count)
    }
}

fn modes_of<const D: usize>(
    mut tangent: NodalStiffnessesSolid<D>,
    nodes: usize,
    masses: &impl InertialMass<D>,
    fixed: &[usize],
    near: Quantity<Rate>,
    count: usize,
) -> Result<Modes<D>, ModalError> {
    let shift = near.value().powi(2);
    masses.add_to_stiffnesses(-shift, &mut tangent);
    let factors = FreeFactored::<D>::factor_blocks(&tangent, fixed)
        .map_err(|error| ModalError::upstream(error, "the shifted stiffnesses"))?;
    let free = factors.free();
    let total = D * nodes;
    let expand = |reduced: &Vector| {
        let mut full = Vector::zero(total);
        free.iter()
            .zip(reduced.as_slice())
            .for_each(|(&index, &entry)| full[index] = entry);
        full
    };
    let gather = |full: &Vector| -> Vector { free.iter().map(|&index| full[index]).collect() };
    let pairs = Lanczos::default()
        .solve(
            |reduced| gather(&factors.solve(&expand(reduced))),
            |reduced| {
                gather(
                    &masses
                        .inertial_forces(&expand(reduced).into())
                        .into_erased()
                        .into(),
                )
            },
            shift,
            free.len(),
            count,
        )
        .map_err(|error| ModalError::upstream(error, "the eigensolver"))?;
    let norm = stiffness_norm(&tangent);
    for (&value, reduced) in pairs.values.iter().zip(&pairs.vectors) {
        let full = expand(reduced);
        let kx = stiffness_times(&tangent, &full);
        let mx: Vector = masses
            .inertial_forces(&full.clone().into())
            .into_erased()
            .into();
        let residual = free
            .iter()
            .map(|&index| (kx[index] - (value - shift) * mx[index]).abs())
            .fold(0.0_f64, f64::max);
        let size = full.as_slice().iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        if residual.is_nan() || residual > VERIFIED * norm * size {
            return Err(ModalError::Unverified(residual / (norm * size)));
        }
    }
    let scale = pairs.values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    if let Some(&value) = pairs.values.iter().find(|&&v| v < -1e-8 * scale) {
        return Err(ModalError::Indefinite(value));
    }
    Ok(Modes {
        frequencies: pairs
            .values
            .iter()
            .map(|&value| Quantity::new(value.max(0.0).sqrt()))
            .collect(),
        shapes: pairs
            .vectors
            .iter()
            .map(|reduced| expand(reduced).into())
            .collect(),
    })
}

const VERIFIED: Scalar = 1e-6;

fn stiffness_times<const D: usize>(tangent: &NodalStiffnessesSolid<D>, x: &Vector) -> Vector {
    let mut product = Vector::zero(x.len());
    tangent.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(b, block)| {
            (0..D).for_each(|i| {
                (0..D).for_each(|j| product[D * a + i] += block[i][j].value() * x[D * b + j])
            })
        })
    });
    product
}

fn stiffness_norm<const D: usize>(tangent: &NodalStiffnessesSolid<D>) -> Scalar {
    let mut sums = vec![0.0; D * tangent.len()];
    tangent.iter().enumerate().for_each(|(a, row)| {
        row.entries().for_each(|(_, block)| {
            (0..D).for_each(|i| (0..D).for_each(|j| sums[D * a + i] += block[i][j].value().abs()))
        })
    });
    sums.into_iter().fold(0.0, Scalar::max)
}

/// Possible errors encountered finding the modes of a solid.
pub enum ModalError {
    /// The stiffness is not positive definite, so there is a mode of negative squared frequency.
    Indefinite(Scalar),
    /// A mode did not satisfy the eigenproblem, as happens when the shift is on a natural frequency.
    Unverified(Scalar),
    Upstream(String, String),
}

impl ModalError {
    fn upstream(error: impl Display, context: &str) -> Self {
        Self::Upstream(error.to_string(), context.to_string())
    }
}

impl StyledError for ModalError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::Indefinite(value) => format!(
                "{h}The solid has a mode of negative squared frequency ({value:?}).{c}\n\
                The stiffness is not positive definite about this state."
            ),
            Self::Unverified(residual) => format!(
                "{h}A mode does not satisfy the eigenproblem (relative residual {residual:e}).{c}\n\
                The frequency to search near may be a natural frequency of the solid."
            ),
            Self::Upstream(error, context) => format!("{error}{c}\nIn {context}."),
        }
    }
}

styled_error!(ModalError);

impl From<ModalError> for AssertionError {
    fn from(error: ModalError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}
