use crate::{
    domain::{
        Model, NodalAccelerations, NodalCoordinates,
        factor::FreeFactored,
        solid::{elastic::ElasticElements, mass::InertialMass},
    },
    math::{
        Current, Quantity, Scalar, Style, StyledError, Tensor, TensorRank1Vec, Vector,
        assert::AssertionError, optimize::Lanczos, styled_error,
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
        let mut tangent = self
            .nodal_stiffnesses(nodal_coordinates)
            .map_err(|error| ModalError::upstream(error, "the stiffnesses"))?;
        let shift = near.value().powi(2);
        masses.add_to_stiffnesses(-shift, &mut tangent);
        let factors = FreeFactored::<D>::factor_blocks(&tangent, fixed)
            .map_err(|error| ModalError::upstream(error, "the shifted stiffnesses"))?;
        let free = factors.free();
        let total = D * nodal_coordinates.len();
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
                    let accelerations = NodalAccelerations::<D>::from(expand(reduced));
                    gather(&masses.inertial_forces(&accelerations).into_erased().into())
                },
                shift,
                free.len(),
                count,
            )
            .map_err(|error| ModalError::upstream(error, "the eigensolver"))?;
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
}

/// Possible errors encountered finding the modes of a solid.
pub enum ModalError {
    /// The stiffness is not positive definite, so there is a mode of negative squared frequency.
    Indefinite(Scalar),
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
