#[cfg(test)]
mod test;

use crate::{
    domain::{
        Blocks, ElementModel, Model, NodalAccelerations, NodalReferenceCoordinates,
        NodalVelocities, block::element::Elements, solid::NodalForcesSolid,
    },
    math::{
        Quantity, QuantitySparseVec2D, QuantityVector, Tensor, Vector,
        sparse::{CscLdl, CscMatrix, SparseError},
    },
    units::{Energy, Mass},
};

pub type NodalMasses = QuantitySparseVec2D<Mass>;
pub type NodalLumpedMasses = QuantityVector<Mass>;

pub trait ConsistentMassElements
where
    Self: Elements,
{
    fn nodal_masses_into(&self, nodal_masses: &mut NodalMasses);
}

pub trait LumpedMassElements<const D: usize>
where
    Self: Elements,
{
    fn nodal_lumped_masses_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    );
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

impl<B1, B2, const D: usize> LumpedMassElements<D> for Blocks<B1, B2>
where
    B1: LumpedMassElements<D>,
    B2: LumpedMassElements<D>,
{
    fn nodal_lumped_masses_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    ) {
        self.0
            .nodal_lumped_masses_into(reference_coordinates, nodal_lumped_masses);
        self.1
            .nodal_lumped_masses_into(reference_coordinates, nodal_lumped_masses)
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

impl<B, const D: usize> LumpedMassElements<D> for Model<B, D>
where
    B: LumpedMassElements<D>,
{
    fn nodal_lumped_masses_into(
        &self,
        reference_coordinates: &NodalReferenceCoordinates<D>,
        nodal_lumped_masses: &mut NodalLumpedMasses,
    ) {
        self.blocks
            .nodal_lumped_masses_into(reference_coordinates, nodal_lumped_masses)
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
    B: LumpedMassElements<D>,
{
    pub fn nodal_lumped_masses(&self) -> NodalLumpedMasses {
        let mut nodal_lumped_masses = NodalLumpedMasses::zero(self.coordinates().len());
        self.nodal_lumped_masses_into(self.coordinates(), &mut nodal_lumped_masses);
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

pub struct FactoredMasses<const D: usize>(CscLdl);

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

/// A mass that yields its inverse with some degrees of freedom held fixed.
///
/// Degrees of freedom are indexed as `D * node + component`. The accelerations of the
/// inverse vanish on the fixed degrees of freedom, and on the others they account for
/// the fixed ones not accelerating.
pub trait MassMatrix<const D: usize> {
    type Inverse: InverseMass<D>;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError>;
}

/// The inverse of lumped masses with fixed degrees of freedom.
pub struct FixedLumpedMasses {
    masses: NodalLumpedMasses,
    fixed: Vec<usize>,
}

impl<const D: usize> MassMatrix<D> for NodalLumpedMasses {
    type Inverse = FixedLumpedMasses;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError> {
        Ok(FixedLumpedMasses {
            masses: self.clone(),
            fixed: fixed.to_vec(),
        })
    }
}

impl<const D: usize> InverseMass<D> for FixedLumpedMasses {
    fn nodal_accelerations(
        &self,
        external_forces: &NodalForcesSolid<D>,
        internal_forces: &NodalForcesSolid<D>,
    ) -> NodalAccelerations<D> {
        let mut accelerations =
            InverseMass::<D>::nodal_accelerations(&self.masses, external_forces, internal_forces);
        self.fixed
            .iter()
            .for_each(|&index| accelerations[index / D][index % D] = Default::default());
        accelerations
    }
}

/// The inverse of a consistent mass restricted to the free degrees of freedom.
pub struct FreeFactoredMasses<const D: usize> {
    factors: CscLdl,
    free: Vec<usize>,
}

impl<const D: usize> MassMatrix<D> for NodalMasses {
    type Inverse = FreeFactoredMasses<D>;
    fn inverse(&self, fixed: &[usize]) -> Result<Self::Inverse, SparseError> {
        let (factors, free) = self.factor_free::<D>(fixed)?;
        Ok(FreeFactoredMasses { factors, free })
    }
}

impl<const D: usize> InverseMass<D> for FreeFactoredMasses<D> {
    fn nodal_accelerations(
        &self,
        external_forces: &NodalForcesSolid<D>,
        internal_forces: &NodalForcesSolid<D>,
    ) -> NodalAccelerations<D> {
        let forces: Vector = (external_forces - internal_forces).into_erased().into();
        let free_forces: Vector = self.free.iter().map(|&index| forces[index]).collect();
        let free_accelerations = self.factors.solve(&free_forces);
        let mut accelerations = Vector::zero(forces.len());
        self.free
            .iter()
            .enumerate()
            .for_each(|(k, &index)| accelerations[index] = free_accelerations[k]);
        accelerations.into()
    }
}

impl NodalMasses {
    pub fn factor<const D: usize>(&self) -> Result<FactoredMasses<D>, SparseError> {
        Ok(FactoredMasses(self.factor_free::<D>(&[])?.0))
    }
    /// Factors the mass restricted to the degrees of freedom that are not fixed,
    /// returning the factors and the indices of the free degrees of freedom.
    fn factor_free<const D: usize>(
        &self,
        fixed: &[usize],
    ) -> Result<(CscLdl, Vec<usize>), SparseError> {
        let mut is_free = vec![true; D * self.len()];
        fixed.iter().for_each(|&index| is_free[index] = false);
        let free: Vec<usize> = (0..D * self.len()).filter(|&i| is_free[i]).collect();
        let mut reduced = vec![usize::MAX; D * self.len()];
        free.iter()
            .enumerate()
            .for_each(|(k, &index)| reduced[index] = k);
        let mut matrix = CscMatrix::from_pattern(
            free.len(),
            free.len(),
            self.iter()
                .enumerate()
                .flat_map(|(a, row)| {
                    row.entries()
                        .flat_map(move |(b, _)| (0..D).map(move |i| (D * a + i, D * b + i)))
                })
                .filter(|&(row, column)| is_free[row] && is_free[column])
                .map(|(row, column)| (reduced[row], reduced[column]))
                .collect(),
        );
        matrix.fill(|row, column| self[free[row] / D][free[column] / D].value());
        let mut factors = matrix.ldl_symbolic()?;
        factors.refactor(&matrix)?;
        Ok((factors, free))
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
