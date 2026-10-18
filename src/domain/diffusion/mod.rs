use crate::{
    constitutive::diffusion::Diffusive,
    domain::{
        Blocks, ElementModel, ElementModelError, Minimize, Model, NodalState, ProvidesTangent,
        Root, SolverFor,
        block::{element::Elements, finalize_node_neighbors, solver_from_neighbors},
        boundary::Constrain,
    },
    math::{
        Quantity, QuantitySparseVec2D, QuantityVector, Tensor,
        optimize::{NewtonRaphson, Optimization, OptimizationError, RootFinding},
    },
};

pub type NodalFields<L> = QuantityVector<<L as Diffusive>::Field>;
pub type NodalForces<L> = QuantityVector<L>;
pub type NodalStiffnesses<L> = QuantitySparseVec2D<<L as Diffusive>::Stiffness>;

pub trait DiffusionElements<L>
where
    L: Diffusive,
    Self: Elements,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, ElementModelError>;
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_forces: &mut NodalForces<L>,
    ) -> Result<(), ElementModelError>;
    fn nodal_forces(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<NodalForces<L>, ElementModelError> {
        let mut nodal_forces = NodalForces::<L>::zero(nodal_fields.len());
        self.nodal_forces_into(nodal_fields, &mut nodal_forces)?;
        Ok(nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_stiffnesses: &mut NodalStiffnesses<L>,
    ) -> Result<(), ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<NodalStiffnesses<L>, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnesses::<L>::zero(nodal_fields.len());
        self.nodal_stiffnesses_into(nodal_fields, &mut nodal_stiffnesses)?;
        Ok(nodal_stiffnesses)
    }
}

impl<L, B, const D: usize> DiffusionElements<L> for Model<B, D>
where
    L: Diffusive,
    B: DiffusionElements<L>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, ElementModelError> {
        self.blocks.potential(nodal_fields)
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_forces: &mut NodalForces<L>,
    ) -> Result<(), ElementModelError> {
        self.blocks.nodal_forces_into(nodal_fields, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_stiffnesses: &mut NodalStiffnesses<L>,
    ) -> Result<(), ElementModelError> {
        self.blocks
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)
    }
}

impl<L, B1, B2> DiffusionElements<L> for Blocks<B1, B2>
where
    L: Diffusive,
    B1: DiffusionElements<L>,
    B2: DiffusionElements<L>,
{
    fn potential(
        &self,
        nodal_fields: &NodalFields<L>,
    ) -> Result<Quantity<L::Energy>, ElementModelError> {
        Ok(self.0.potential(nodal_fields)? + self.1.potential(nodal_fields)?)
    }
    fn nodal_forces_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_forces: &mut NodalForces<L>,
    ) -> Result<(), ElementModelError> {
        self.0.nodal_forces_into(nodal_fields, nodal_forces)?;
        self.1.nodal_forces_into(nodal_fields, nodal_forces)
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_fields: &NodalFields<L>,
        nodal_stiffnesses: &mut NodalStiffnesses<L>,
    ) -> Result<(), ElementModelError> {
        self.0
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)?;
        self.1
            .nodal_stiffnesses_into(nodal_fields, nodal_stiffnesses)
    }
}

impl<L, B, const D: usize> SolverFor<Model<B, D>, NodalForces<L>, NodalStiffnesses<L>>
    for NewtonRaphson
where
    L: Diffusive,
    B: DiffusionElements<L>,
{
    type Tangent = NodalStiffnesses<L>;
    const SPARSE: bool = true;
}

impl<L, B, const D: usize> Root<NodalForces<L>, NodalStiffnesses<L>, NodalFields<L>> for Model<B, D>
where
    L: Diffusive,
    B: DiffusionElements<L>,
{
    fn root<S>(
        &self,
        boundary_conditions: &impl Constrain<<NodalFields<L> as NodalState>::Layout>,
        solver: S,
    ) -> Result<NodalFields<L>, OptimizationError>
    where
        S: SolverFor<Self, NodalForces<L>, NodalStiffnesses<L>>
            + RootFinding<NodalForces<L>, S::Tangent, NodalFields<L>>,
        Self: ProvidesTangent<NodalFields<L>, S::Tangent>,
    {
        let equality_constraint = boundary_conditions.constraint();
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.root(
            |nodal_fields: &NodalFields<L>| Ok(self.nodal_forces(nodal_fields)?),
            |nodal_fields: &NodalFields<L>| Ok(self.provide_tangent(nodal_fields)?),
            NodalFields::<L>::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}

impl<L, B, const D: usize> SolverFor<Model<B, D>, Quantity<L::Energy>, NodalForces<L>>
    for NewtonRaphson
where
    L: Diffusive,
    B: DiffusionElements<L>,
{
    type Tangent = NodalStiffnesses<L>;
    const SPARSE: bool = true;
}

impl<L, B, const D: usize> Minimize<Quantity<L::Energy>, NodalForces<L>, NodalFields<L>>
    for Model<B, D>
where
    L: Diffusive,
    B: DiffusionElements<L>,
{
    fn minimize<S>(
        &self,
        boundary_conditions: &impl Constrain<<NodalFields<L> as NodalState>::Layout>,
        solver: S,
    ) -> Result<NodalFields<L>, OptimizationError>
    where
        S: SolverFor<Self, Quantity<L::Energy>, NodalForces<L>>
            + Optimization<Quantity<L::Energy>, NodalForces<L>, S::Tangent, NodalFields<L>>,
        Self: ProvidesTangent<NodalFields<L>, S::Tangent>,
    {
        let equality_constraint = boundary_conditions.constraint();
        let sparse = S::SPARSE.then(|| {
            let mut neighbors = vec![Vec::new(); self.coordinates().len()];
            self.node_neighbors(&mut neighbors);
            finalize_node_neighbors(&mut neighbors);
            solver_from_neighbors(&neighbors, &equality_constraint, 1, true)
        });
        solver.minimize(
            |nodal_fields: &NodalFields<L>| Ok(self.potential(nodal_fields)?),
            |nodal_fields: &NodalFields<L>| Ok(self.nodal_forces(nodal_fields)?),
            |nodal_fields: &NodalFields<L>| Ok(self.provide_tangent(nodal_fields)?),
            NodalFields::<L>::zero(self.coordinates().len()),
            equality_constraint,
            sparse,
        )
    }
}
