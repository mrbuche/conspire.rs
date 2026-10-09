use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        factor::fixed_indices,
        solid::{
            NodalForcesSolid,
            elastic::ElasticElements,
            mass::{InverseMass, MassMatrix, NodalLumpedMasses},
            time_scale::TimeScaleElements,
        },
    },
    math::{
        Quantity, Scalar,
        integrate::{ExplicitDynamics, IntegrationError, Times},
        optimize::EqualityConstraint,
    },
    units::Time,
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
);

fn held<M>(
    masses: &M,
    equality_constraint: EqualityConstraint,
    mut velocities: NodalVelocities<3>,
) -> Result<(M::Inverse, NodalVelocities<3>), IntegrationError>
where
    M: MassMatrix<3>,
{
    let fixed = fixed_indices(equality_constraint, "explicit dynamics")?;
    let inverse = masses
        .inverse(&fixed)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    fixed
        .iter()
        .for_each(|&index| velocities[index / 3][index % 3] = Default::default());
    Ok((inverse, velocities))
}

/// The motion of an elastic model.
pub trait ElasticDynamics {
    fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError>;
    /// Integrates the motion of the model with an explicit dynamics integrator.
    ///
    /// The masses may be lumped or consistent. Fixed degrees of freedom, whose indices are
    /// `D * node + component`, are held by giving them no velocity, and no acceleration
    /// from the inverse of the masses. Linear constraints are not supported.
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &impl MassMatrix<3>,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>;
    /// Integrates the motion of the model like [`integrate`](Self::integrate), with a time
    /// step limited by the fastest time scale of the elements.
    ///
    /// Only lumped masses are accepted, since the estimate of the highest frequency is that of
    /// the lumped masses, which a consistent mass exceeds. The estimate assembles the element
    /// stiffnesses, and is refreshed every `interval` evaluations of the bound, which for the
    /// integrators here is every step. The time step may use at most the fraction `safety` of
    /// the stability limit, and a step above it is an error.
    #[expect(clippy::too_many_arguments)]
    fn integrate_bounded(
        &self,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &NodalLumpedMasses,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>
    where
        Self: TimeScaleElements<3>;
}

impl<B> ElasticDynamics for Model<B, 3>
where
    B: ElasticElements<3>,
{
    fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        Ok(masses.nodal_accelerations(external_forces, &self.nodal_forces(nodal_coordinates)?))
    }
    fn integrate(
        &self,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &impl MassMatrix<3>,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError> {
        let (masses, initial_velocities) = held(masses, equality_constraint, initial_velocities)?;
        integrator.integrate(
            |_, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                self.nodal_accelerations(coordinates, external_forces, &masses)
                    .map_err(|error| error.to_string())
            },
            time,
            initial_coordinates,
            initial_velocities,
        )
    }
    fn integrate_bounded(
        &self,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        safety: Scalar,
        interval: usize,
        time: &[Quantity<Time>],
        (initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &NodalLumpedMasses,
        equality_constraint: EqualityConstraint,
    ) -> Result<Solution, IntegrationError>
    where
        Self: TimeScaleElements<3>,
    {
        if interval == 0 {
            return Err(IntegrationError::Intermediate(
                "The interval between estimates of the time scale must be at least one."
                    .to_string(),
            ));
        }
        let (masses, initial_velocities) = held(masses, equality_constraint, initial_velocities)?;
        let mut evaluations = 0;
        let mut time_scale = Time::seconds(Scalar::INFINITY);
        integrator.integrate_bounded(
            |_, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                self.nodal_accelerations(coordinates, external_forces, &masses)
                    .map_err(|error| error.to_string())
            },
            |_, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                if evaluations % interval == 0 {
                    time_scale = self
                        .fastest_time_scale(&self.coordinates, coordinates)
                        .map_err(|error| error.to_string())?;
                }
                evaluations += 1;
                Ok(time_scale)
            },
            safety,
            time,
            initial_coordinates,
            initial_velocities,
        )
    }
}
