use crate::{
    domain::{
        ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        mass::{InverseMass, MassMatrix},
        solid::{NodalForcesSolid, elastic::ElasticElements},
    },
    math::{
        Quantity,
        integrate::{ExplicitDynamics, IntegrationError, Times},
        optimize::EqualityConstraint,
    },
    units::Time,
};

impl<B> Model<B, 3>
where
    B: ElasticElements<3>,
{
    pub fn nodal_accelerations(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        external_forces: &NodalForcesSolid<3>,
        masses: &impl InverseMass<3>,
    ) -> Result<NodalAccelerations<3>, ElementModelError> {
        Ok(masses.nodal_accelerations(external_forces, &self.nodal_forces(nodal_coordinates)?))
    }
    /// Integrates the motion of the model with an explicit dynamics integrator.
    ///
    /// The masses may be lumped or consistent. Fixed degrees of freedom, whose indices are
    /// `3 * node + component`, are held by giving them no velocity, and no acceleration
    /// from the inverse of the masses. Linear constraints are not supported.
    pub fn integrate(
        &self,
        integrator: &impl ExplicitDynamics<
            NodalCoordinates<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        time: &[Quantity<Time>],
        (initial_coordinates, mut initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        external_forces: &NodalForcesSolid<3>,
        masses: &impl MassMatrix<3>,
        equality_constraint: EqualityConstraint,
    ) -> Result<
        (
            Times,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        ),
        IntegrationError,
    > {
        let fixed = match equality_constraint {
            EqualityConstraint::Fixed(indices) => indices,
            EqualityConstraint::None => vec![],
            EqualityConstraint::Linear(..) => {
                return Err(IntegrationError::Intermediate(
                    "Linear constraints are not supported by explicit dynamics.".to_string(),
                ));
            }
        };
        let masses = masses
            .inverse(&fixed)
            .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
        fixed
            .iter()
            .for_each(|&index| initial_velocities[index / 3][index % 3] = Default::default());
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
}
