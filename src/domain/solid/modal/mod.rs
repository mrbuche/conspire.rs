#[cfg(test)]
pub(crate) mod test;

use crate::{
    domain::{
        Model, NodalCoordinates, NodalVelocities,
        factor::FreeFactored,
        solid::{
            NodalStiffnessesSolid, elastic::ElasticElements,
            elastic_plastic::ElasticPlasticElements,
            elastic_viscoplastic::ElasticViscoplasticElements, hyperelastic::HyperelasticElements,
            hyperelastic_plastic::HyperelasticPlasticElements,
            hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
            hyperviscoelastic::HyperviscoelasticElements, mass::InertialMass,
            viscoelastic::ViscoelasticElements,
        },
    },
    math::{
        Current, Differentiable, Quantity, Scalar, Style, StyledError, Tensor, TensorRank1Vec,
        TensorRank2SparseVec2D, Vector, assert::AssertionError, optimize::Lanczos, styled_error,
    },
    units::Rate,
};
use core::fmt::Display;

/// The natural frequencies of a solid and the shapes it vibrates in.
#[derive(Clone, Debug)]
pub struct Modes<const D: usize> {
    /// Angular frequencies, ascending.
    pub frequencies: Vec<Quantity<Rate>>,
    /// Shapes with zero entries at the fixed degrees of freedom, normalized
    /// so that the mass they carry is one in SI units.
    pub shapes: Vec<TensorRank1Vec<D, Current>>,
}

/// Vibration of hyperelastic solids about a given state.
pub trait HyperelasticModes<const D: usize> {
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

impl<B, const D: usize> HyperelasticModes<D> for Model<B, D>
where
    B: HyperelasticElements<D>,
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

/// Vibration of hyperelastic-viscoplastic solids about a given state.
pub trait HyperelasticViscoplasticModes<S, const D: usize> {
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

impl<B, S, const D: usize> HyperelasticViscoplasticModes<S, D> for Model<B, D>
where
    B: HyperelasticViscoplasticElements<S, D>,
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

/// The natural frequencies of a viscoelastic solid, and the damping of each.
#[derive(Clone, Debug)]
pub struct DampedModes<const D: usize> {
    /// The undamped modes of the stiffness at rest.
    pub modes: Modes<D>,
    /// The fraction of critical damping of each mode, taken as zero for a mode of zero frequency.
    pub damping_ratios: Vec<Scalar>,
}

impl<const D: usize> DampedModes<D> {
    /// The frequencies of the damped oscillation, zero for a mode that is critically
    /// damped or more.
    pub fn damped_frequencies(&self) -> Vec<Quantity<Rate>> {
        self.modes
            .frequencies
            .iter()
            .zip(&self.damping_ratios)
            .map(|(frequency, ratio)| {
                Quantity::new(frequency.value() * (1.0 - ratio * ratio).max(0.0).sqrt())
            })
            .collect()
    }
}

/// Vibration of hyperviscoelastic solids about a given state.
pub trait HyperviscoelasticModes<const D: usize> {
    /// The `count` modes with angular frequencies nearest `near`, of small vibrations
    /// about the given coordinates, with the damping of each.
    ///
    /// The frequencies and shapes are those of the stiffness at rest, with the
    /// deformation gradient rate zero, and the mass. The damping ratio of a mode is
    /// $`\varphi^T C \varphi / 2\omega`$ for its mass-normalized shape $`\varphi`$ and
    /// the damping matrix $`C`$ at rest. That is exact when the damping is proportional
    /// to the stiffness or to the mass, as it is for the axial modes of a bar, and
    /// otherwise a first-order estimate that ignores the coupling of modes by the damping.
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<DampedModes<D>, ModalError>;
}

impl<B, const D: usize> HyperviscoelasticModes<D> for Model<B, D>
where
    B: HyperviscoelasticElements<D>,
{
    fn modes(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        masses: &impl InertialMass<D>,
        fixed: &[usize],
        near: Quantity<Rate>,
        count: usize,
    ) -> Result<DampedModes<D>, ModalError> {
        let at_rest = NodalVelocities::<D>::zero(nodal_coordinates.len());
        let tangent = self
            .nodal_stiffnesses(nodal_coordinates, &at_rest)
            .map_err(|error| ModalError::upstream(error, "the stiffnesses"))?;
        let modes = modes_of(tangent, nodal_coordinates.len(), masses, fixed, near, count)?;
        let dampings = self
            .nodal_dampings(nodal_coordinates, &at_rest)
            .map_err(|error| ModalError::upstream(error, "the dampings"))?;
        let damping_ratios = modes
            .frequencies
            .iter()
            .zip(&modes.shapes)
            .map(|(frequency, shape)| {
                let shape: Vector = shape.clone().into_erased().into();
                let rate = &shape * &block_times(&dampings, &shape);
                let frequency = frequency.value();
                if frequency > 0.0 {
                    rate / (2.0 * frequency)
                } else {
                    0.0
                }
            })
            .collect();
        Ok(DampedModes {
            modes,
            damping_ratios,
        })
    }
}

/// Vibration of hyperelastic-plastic solids about a given state.
pub trait HyperelasticPlasticModes<S, const D: usize> {
    /// The `count` modes with angular frequencies nearest `near`, of small
    /// vibrations about the given coordinates and plastic state.
    ///
    /// The plastic state is the one at the coordinates, as committed by the stepper
    /// after a load step. The stiffness is the elastic tangent with the plastic
    /// deformation gradient held, and not the consistent tangent of a plastic step:
    /// a vibration unloads as much as it loads, and unloading is elastic, so the
    /// vibration shakes down to an elastic response about the state.
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

impl<B, S, const D: usize> HyperelasticPlasticModes<S, D> for Model<B, D>
where
    B: HyperelasticPlasticElements<S, D>,
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
            .nodal_elastic_stiffnesses(nodal_coordinates, state_variables)
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
        let kx = block_times(&tangent, &full);
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

fn block_times<const D: usize, U>(
    tangent: &TensorRank2SparseVec2D<D, Current, Current, U>,
    x: &Vector,
) -> Vector {
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
