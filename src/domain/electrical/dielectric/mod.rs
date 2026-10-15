use crate::{
    domain::{
        Blocks, ElementModel, ElementModelError, Minimize, Model, ProvidesTangent, Root, SolverFor,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        electrical::{NodalForcesElectrical, NodalPotentials, NodalStiffnessesElectrical},
    },
    math::{
        Quantity, Tensor,
        optimize::{
            EqualityConstraint, NewtonRaphson, Optimization, OptimizationError, RootFinding,
        },
    },
    units::Energy,
};

pub trait DielectricElements
where
    Self: Elements,
{
    fn potential(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<Quantity<Energy>, ElementModelError>;
    fn nodal_forces_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_forces: &mut NodalForcesElectrical,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<NodalForcesElectrical, ElementModelError> {
        let mut nodal_forces = NodalForcesElectrical::zero(nodal_potentials.len());
        self.nodal_forces_into(nodal_potentials, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_stiffnesses: &mut NodalStiffnessesElectrical,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<NodalStiffnessesElectrical, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesElectrical::zero(nodal_potentials.len());
        self.nodal_stiffnesses_into(nodal_potentials, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
}

impl<B, const D: usize> DielectricElements for Model<B, D>
where
    B: DielectricElements,
{
    fn potential(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.blocks.potential(nodal_potentials)
    }
    fn nodal_forces_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_forces: &mut NodalForcesElectrical,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_forces_into(nodal_potentials, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_stiffnesses: &mut NodalStiffnessesElectrical,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_potentials, nodal_stiffnesses)
    }
}

impl<B1, B2> DielectricElements for Blocks<B1, B2>
where
    B1: DielectricElements,
    B2: DielectricElements,
{
    fn potential(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        Ok(self.0.potential(nodal_potentials)? + self.1.potential(nodal_potentials)?)
    }
    fn nodal_forces_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_forces: &mut NodalForcesElectrical,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_potentials, nodal_forces)?;
        self.1.nodal_forces_into(nodal_potentials, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_potentials: &NodalPotentials,
        nodal_stiffnesses: &mut NodalStiffnessesElectrical,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_potentials, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_potentials, nodal_stiffnesses)
    }
}

impl<B, const D: usize> ProvidesTangent<NodalPotentials, NodalStiffnessesElectrical> for Model<B, D>
where
    B: DielectricElements,
{
    fn provide_tangent(
        &self,
        nodal_potentials: &NodalPotentials,
    ) -> Result<NodalStiffnessesElectrical, ElementModelError> {
        self.nodal_stiffnesses(nodal_potentials)
    }
}

impl<B, const D: usize> SolverFor<Model<B, D>, NodalForcesElectrical, NodalStiffnessesElectrical>
    for NewtonRaphson
where
    B: DielectricElements,
{
    type Tangent = NodalStiffnessesElectrical;
    const SPARSE: bool = true;
}

impl<B, const D: usize> Root<NodalForcesElectrical, NodalStiffnessesElectrical, NodalPotentials>
    for Model<B, D>
where
    B: DielectricElements,
{
    fn root<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalPotentials, OptimizationError>
    where
        S: SolverFor<Self, NodalForcesElectrical, NodalStiffnessesElectrical>
            + RootFinding<NodalForcesElectrical, S::Tangent, NodalPotentials>,
        Self: ProvidesTangent<NodalPotentials, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.root(
            |nodal_potentials: &NodalPotentials| Ok(self.nodal_forces(nodal_potentials)?),
            |nodal_potentials: &NodalPotentials| Ok(self.provide_tangent(nodal_potentials)?),
            NodalPotentials::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}

impl<B, const D: usize> SolverFor<Model<B, D>, Quantity<Energy>, NodalForcesElectrical>
    for NewtonRaphson
where
    B: DielectricElements,
{
    type Tangent = NodalStiffnessesElectrical;
    const SPARSE: bool = true;
}

impl<B, const D: usize> Minimize<Quantity<Energy>, NodalForcesElectrical, NodalPotentials>
    for Model<B, D>
where
    B: DielectricElements,
{
    fn minimize<S>(
        &self,
        equality_constraint: EqualityConstraint,
        solver: S,
    ) -> Result<NodalPotentials, OptimizationError>
    where
        S: SolverFor<Self, Quantity<Energy>, NodalForcesElectrical>
            + Optimization<Quantity<Energy>, NodalForcesElectrical, S::Tangent, NodalPotentials>,
        Self: ProvidesTangent<NodalPotentials, S::Tangent>,
    {
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.minimize(
            |nodal_potentials: &NodalPotentials| Ok(self.potential(nodal_potentials)?),
            |nodal_potentials: &NodalPotentials| Ok(self.nodal_forces(nodal_potentials)?),
            |nodal_potentials: &NodalPotentials| Ok(self.provide_tangent(nodal_potentials)?),
            NodalPotentials::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}
