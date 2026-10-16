use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        Blocks, ElementModel, ElementModelError, Minimize, Model, ProvidesTangent, Root, SolverFor,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
    },
    math::{
        Quantity, QuantitySparseVec2D, QuantityVector, Tensor,
        optimize::{EqualityConstraint, Optimization, OptimizationError, RootFinding},
    },
};

pub type NodalFields<U> = QuantityVector<U>;
pub type NodalForces<U> = QuantityVector<<U as Diffusive>::Force>;
pub type NodalStiffnesses<U> = QuantitySparseVec2D<<U as Diffusive>::Stiffness>;

pub trait DiffusionElements<U>
where
    U: Diffusive,
    Self: Elements,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<Quantity<U::Energy>, ElementModelError>;
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_forces: &mut NodalForces<U>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<NodalForces<U>, ElementModelError> {
        let mut nodal_forces = NodalForces::<U>::zero(nodal_fields.len());
        self.nodal_forces_into(nodal_fields, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_stiffnesses: &mut NodalStiffnesses<U>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<NodalStiffnesses<U>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnesses::<U>::zero(nodal_fields.len());
        self.nodal_stiffnesses_into(nodal_fields, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
}

impl<U, B, const D: usize> DiffusionElements<U> for Model<B, D>
where
    U: Diffusive,
    B: DiffusionElements<U>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<Quantity<U::Energy>, ElementModelError> {
        self.blocks.potential(nodal_fields)
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_forces: &mut NodalForces<U>,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_forces_into(nodal_fields, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_stiffnesses: &mut NodalStiffnesses<U>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)
    }
}

impl<U, B1, B2> DiffusionElements<U> for Blocks<B1, B2>
where
    U: Diffusive,
    B1: DiffusionElements<U>,
    B2: DiffusionElements<U>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<Quantity<U::Energy>, ElementModelError> {
        Ok(self.0.potential(nodal_fields)? + self.1.potential(nodal_fields)?)
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_forces: &mut NodalForces<U>,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_fields, nodal_forces)?;
        self.1.nodal_forces_into(nodal_fields, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<U>,
        nodal_stiffnesses: &mut NodalStiffnesses<U>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)
    }
}

impl<U, B, const D: usize> ProvidesTangent<NodalFields<U>, NodalStiffnesses<U>> for Model<B, D>
where
    U: Diffusive,
    B: DiffusionElements<U>,
{
    fn provide_tangent(
        &self,
        nodal_fields: &NodalFields<U>,
    ) -> Result<NodalStiffnesses<U>, ElementModelError> {
        self.nodal_stiffnesses(nodal_fields)
    }
}

impl<U, B, const D: usize> Root<NodalForces<U>, NodalStiffnesses<U>, NodalFields<U>> for Model<B, D>
where
    U: Diffusive,
    B: DiffusionElements<U>,
{
    fn root<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalFields<U>, OptimizationError>
    where
        S: SolverFor<Self, NodalForces<U>, NodalStiffnesses<U>>
            + RootFinding<NodalForces<U>, S::Tangent, NodalFields<U>>,
        Self: ProvidesTangent<NodalFields<U>, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.root(
            |nodal_fields: &NodalFields<U>| Ok(self.nodal_forces(nodal_fields)?),
            |nodal_fields: &NodalFields<U>| Ok(self.provide_tangent(nodal_fields)?),
            NodalFields::<U>::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}

impl<U, B, const D: usize> Minimize<Quantity<U::Energy>, NodalForces<U>, NodalFields<U>>
    for Model<B, D>
where
    U: Diffusive,
    B: DiffusionElements<U>,
{
    fn minimize<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalFields<U>, OptimizationError>
    where
        S: SolverFor<Self, Quantity<U::Energy>, NodalForces<U>>
            + Optimization<Quantity<U::Energy>, NodalForces<U>, S::Tangent, NodalFields<U>>,
        Self: ProvidesTangent<NodalFields<U>, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.minimize(
            |nodal_fields: &NodalFields<U>| Ok(self.potential(nodal_fields)?),
            |nodal_fields: &NodalFields<U>| Ok(self.nodal_forces(nodal_fields)?),
            |nodal_fields: &NodalFields<U>| Ok(self.provide_tangent(nodal_fields)?),
            NodalFields::<U>::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}
