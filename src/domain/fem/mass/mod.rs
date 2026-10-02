use crate::{
    fem::{
        Blocks, ElementModel, Elements, Model, NodalAccelerations, NodalVelocities,
        block::mass::{NodalLumpedMasses, NodalMasses},
        solid::NodalForcesSolid,
    },
    math::{
        Quantity, Tensor,
        sparse::{CscLu, CscMatrix, SparseError},
    },
    units::Energy,
};

pub trait ConsistentMassElements
where
    Self: Elements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses);
}

pub trait LumpedMassElements
where
    Self: Elements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses);
}

impl<B1, B2> ConsistentMassElements for Blocks<B1, B2>
where
    B1: ConsistentMassElements,
    B2: ConsistentMassElements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.0.nodal_masses_into(nodal_masses);
        self.1.nodal_masses_into(nodal_masses)
    }
}

impl<B1, B2> LumpedMassElements for Blocks<B1, B2>
where
    B1: LumpedMassElements,
    B2: LumpedMassElements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses) {
        self.0.nodal_lumped_masses_into(nodal_lumped_masses);
        self.1.nodal_lumped_masses_into(nodal_lumped_masses)
    }
}

impl<B, const D: usize> ConsistentMassElements for Model<B, D>
where
    B: ConsistentMassElements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses) {
        self.blocks.nodal_masses_into(nodal_masses)
    }
}

impl<B, const D: usize> LumpedMassElements for Model<B, D>
where
    B: LumpedMassElements,
{
    fn nodal_lumped_masses_into(&self, nodal_lumped_masses: &mut NodalLumpedMasses) {
        self.blocks.nodal_lumped_masses_into(nodal_lumped_masses)
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: ConsistentMassElements,
{
    pub fn nodal_masses(&self) -> NodalMasses {
        let mut nodal_masses = NodalMasses::zero(self.coordinates().len());
        self.nodal_masses_into(&mut nodal_masses);
        nodal_masses
    }
}

impl<B, const D: usize> Model<B, D>
where
    B: LumpedMassElements,
{
    pub fn nodal_lumped_masses(&self) -> NodalLumpedMasses {
        let mut nodal_lumped_masses = NodalLumpedMasses::zero(self.coordinates().len());
        self.nodal_lumped_masses_into(&mut nodal_lumped_masses);
        nodal_lumped_masses
    }
}

impl NodalLumpedMasses {
    pub fn kinetic_energy<const D: usize>(
        &self,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Quantity<Energy> {
        self.iter()
            .zip(nodal_velocities.iter())
            .map(|(&mass, velocity)| mass * (velocity * velocity))
            .sum::<Quantity<Energy>>()
            * 0.5
    }
    pub fn inertial_forces<const D: usize>(
        &self,
        nodal_accelerations: &NodalAccelerations<D>,
    ) -> NodalForcesSolid<D> {
        self.iter()
            .zip(nodal_accelerations.iter())
            .map(|(&mass, acceleration)| acceleration * mass)
            .collect()
    }
}

pub trait InverseMass<const D: usize> {
    fn nodal_accelerations(
        &self,
        external_forces: &NodalForcesSolid<D>,
        internal_forces: &NodalForcesSolid<D>,
    ) -> NodalAccelerations<D>;
}

impl<const D: usize> InverseMass<D> for NodalLumpedMasses {
    fn nodal_accelerations(
        &self,
        external_forces: &NodalForcesSolid<D>,
        internal_forces: &NodalForcesSolid<D>,
    ) -> NodalAccelerations<D> {
        self.iter()
            .zip(external_forces.iter().zip(internal_forces.iter()))
            .map(|(&mass, (external_force, internal_force))| {
                (external_force - internal_force) / mass
            })
            .collect()
    }
}

pub struct FactoredMasses<const D: usize>(CscLu);

impl<const D: usize> InverseMass<D> for FactoredMasses<D> {
    fn nodal_accelerations(
        &self,
        external_forces: &NodalForcesSolid<D>,
        internal_forces: &NodalForcesSolid<D>,
    ) -> NodalAccelerations<D> {
        self.0
            .solve(&(external_forces - internal_forces).into_erased().into())
            .into()
    }
}

impl NodalMasses {
    pub fn factor<const D: usize>(&self) -> Result<FactoredMasses<D>, SparseError> {
        let mut matrix = CscMatrix::from_pattern(
            D * self.len(),
            D * self.len(),
            self.iter()
                .enumerate()
                .flat_map(|(a, row)| {
                    row.entries()
                        .flat_map(move |(b, _)| (0..D).map(move |i| (D * a + i, D * b + i)))
                })
                .collect(),
        );
        matrix.fill(|row, column| self[row / D][column / D].value());
        Ok(FactoredMasses(matrix.lu_amd()?))
    }
    pub fn kinetic_energy<const D: usize>(
        &self,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Quantity<Energy> {
        self.iter()
            .zip(nodal_velocities.iter())
            .flat_map(|(row, velocity_a)| {
                row.entries()
                    .map(move |(b, &mass)| mass * (velocity_a * &nodal_velocities[b]))
            })
            .sum::<Quantity<Energy>>()
            * 0.5
    }
    pub fn inertial_forces<const D: usize>(
        &self,
        nodal_accelerations: &NodalAccelerations<D>,
    ) -> NodalForcesSolid<D> {
        self.iter()
            .map(|row| {
                row.entries()
                    .map(|(b, &mass)| &nodal_accelerations[b] * mass)
                    .sum()
            })
            .collect()
    }
}
