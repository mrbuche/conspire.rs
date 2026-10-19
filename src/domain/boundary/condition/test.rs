use crate::{
    domain::{
        NodalCoordinates,
        boundary::{
            BoundaryConditions, Constrain, ConstrainAt, Coupled, Direct, Displacement, Field,
            Layout, Left, NodalState, Potential, Right, Solid, SolidThermal, Temperature,
        },
        multiphysics::solid_thermal::NodalStatesSolidThermal,
    },
    geometry::mesh::{NodeSet, NodeSets},
    math::{
        Current, Quantity, QuantityVector, TensorRank1, TensorTuple, TensorTupleVec,
        optimize::{EqualityConstraint, EqualityConstraints},
    },
    units::{ElectricPotential, Length, Temperature as TemperatureUnit},
};
use std::marker::PhantomData;

pub(super) fn indices(constraints: EqualityConstraints) -> Vec<usize> {
    EqualityConstraint::held_indices(&constraints).unwrap_or_default()
}

pub(super) fn sets() -> NodeSets {
    vec![
        NodeSet::new(vec![0, 2]).with_label(7).with_name("bottom"),
        NodeSet::new(vec![1]).with_name("top"),
    ]
    .into()
}

pub(super) fn layout_of<X>() -> PhantomData<X::Layout>
where
    X: NodalState,
{
    PhantomData
}

#[test]
fn nothing_fixed() {
    let fixed = BoundaryConditions::<Solid<3>>::new();
    assert!(indices(fixed.constraint()).is_empty());
}

#[test]
fn a_set_that_is_not_there() {
    assert!(sets().by_name("side").is_none());
    assert!(sets().by_label(8).is_none());
}

mod coupled {
    use super::*;
    type Three = Coupled<Coupled<Solid<3>, Field<TemperatureUnit>>, Field<ElectricPotential>>;
    #[test]
    fn three_physics_are_laid_out_in_order() {
        assert_eq!(Three::STRIDE, 5);
        let fixed = BoundaryConditions::<Three>::new()
            .fix_node(Displacement::<1>, 2)
            .fix_node(Temperature, 2)
            .fix_node(Potential, 2);
        assert_eq!(indices(fixed.constraint()), [11, 13, 14]);
    }
    #[test]
    fn the_order_of_the_physics_is_the_order_of_the_offsets() {
        let fixed = BoundaryConditions::<Coupled<Field<TemperatureUnit>, Solid<3>>>::new()
            .fix_node(Temperature, 1)
            .fix_node(Displacement::<2>, 1);
        assert_eq!(indices(fixed.constraint()), [4, 7]);
    }
    #[test]
    fn a_pair_may_be_nested_on_either_side() {
        let fixed = BoundaryConditions::<
            Coupled<Solid<2>, Coupled<Field<TemperatureUnit>, Field<ElectricPotential>>>,
        >::new()
        .fix_node(Displacement::<1>, 1)
        .fix_node(Temperature, 1)
        .fix_node(Potential, 1);
        assert_eq!(indices(fixed.constraint()), [5, 6, 7]);
    }
    #[test]
    fn the_same_physics_twice_is_told_apart_by_a_marker() {
        let fixed = BoundaryConditions::<Coupled<Solid<3>, Solid<3>>>::new()
            .fix_node::<_, Left<Direct>>(Displacement::<1>, 1)
            .fix_node::<_, Right<Direct>>(Displacement::<1>, 1);
        assert_eq!(indices(fixed.constraint()), [7, 10]);
    }
    #[test]
    fn a_physics_may_be_left_alone() {
        let fixed = BoundaryConditions::<Three>::new().fix_nodes(Potential, 0..3);
        assert_eq!(indices(fixed.constraint()), [4, 9, 14]);
    }
    #[test]
    fn the_state_of_a_node_gives_the_layout() {
        let _: PhantomData<Solid<3>> = layout_of::<NodalCoordinates<3>>();
        let _: PhantomData<SolidThermal<3>> = layout_of::<NodalStatesSolidThermal>();
        let _: PhantomData<Field<TemperatureUnit>> = layout_of::<QuantityVector<TemperatureUnit>>();
    }
    #[test]
    fn a_state_of_new_physics_needs_no_layout_of_its_own() {
        type State = TensorTupleVec<
            TensorTuple<TensorRank1<3, Current, Length>, Quantity<TemperatureUnit>>,
            Quantity<ElectricPotential>,
        >;
        let _: PhantomData<Three> = layout_of::<State>();
    }
}

mod solve_with {
    use crate::{
        domain::{
            ElementModelError, Minimize, NodalCoordinates, ProvidesTangent, Root, SolverFor,
            boundary::{
                Conservative, Constrain, ConstrainAt, Displacement, Load, LoadStiffness,
                NodalState, Temperature,
            },
            multiphysics::solid_thermal::NodalStatesSolidThermal,
        },
        geometry::mesh::NodeSet,
        math::{
            optimize::{
                EqualityConstraint, EqualityConstraints, Optimization, OptimizationError,
                Prescription, RootFinding,
            },
            sparse::SparseSolver,
        },
    };
    struct Recorder;
    struct Dummy<X>(X);
    impl<X> SolverFor<Dummy<X>, (), ()> for Recorder {
        type Tangent = ();
        const SPARSE: bool = false;
    }
    impl<X> RootFinding<(), (), X> for Recorder {
        fn root(
            &self,
            _: impl FnMut(&X) -> Result<(), String>,
            _: impl FnMut(&X) -> Result<(), String>,
            _: X,
            equality_constraint: EqualityConstraints,
            _: Option<SparseSolver>,
        ) -> Result<X, OptimizationError> {
            record(equality_constraint)
        }
    }
    impl<X> Optimization<(), (), (), X> for Recorder {
        fn minimize(
            &self,
            _: impl FnMut(&X) -> Result<(), String>,
            _: impl FnMut(&X) -> Result<(), String>,
            _: impl FnMut(&X) -> Result<(), String>,
            _: X,
            equality_constraint: EqualityConstraints,
            _: Option<SparseSolver>,
        ) -> Result<X, OptimizationError> {
            record(equality_constraint)
        }
    }
    fn record<X>(equality_constraints: EqualityConstraints) -> Result<X, OptimizationError> {
        let merged = EqualityConstraint::merged(equality_constraints);
        match merged.as_slice() {
            [EqualityConstraint::Fixed(indices)] => {
                Err(OptimizationError::Intermediate(format!("{indices:?}")))
            }
            [EqualityConstraint::Prescribed(entries)]
                if entries
                    .iter()
                    .all(|&(_, prescription)| prescription == Prescription::Initial) =>
            {
                let indices: Vec<usize> = entries.iter().map(|&(index, _)| index).collect();
                Err(OptimizationError::Intermediate(format!("{indices:?}")))
            }
            [EqualityConstraint::Prescribed(entries)] => {
                Err(OptimizationError::Intermediate(format!("{entries:?}")))
            }
            _ => panic!("expected a fixed or prescribed constraint"),
        }
    }
    impl<X> ProvidesTangent<X, ()> for Dummy<X> {
        fn provide_tangent(&self, _: &X) -> Result<(), ElementModelError> {
            Ok(())
        }
    }
    impl<X> Root<(), (), X> for Dummy<X>
    where
        X: Clone + NodalState,
    {
        fn root<S>(
            &self,
            boundary_conditions: &impl Constrain<X::Layout>,
            _: &(impl Load<X, ()> + LoadStiffness<X, S::Tangent>),
            solver: S,
        ) -> Result<X, OptimizationError>
        where
            S: SolverFor<Self, (), ()> + RootFinding<(), S::Tangent, X>,
            Self: ProvidesTangent<X, S::Tangent>,
        {
            solver.root(
                |_| Ok(()),
                |x| self.provide_tangent(x).map_err(String::from),
                self.0.clone(),
                boundary_conditions.constraint(),
                None,
            )
        }
    }
    impl<X> Minimize<(), (), X> for Dummy<X>
    where
        X: Clone + NodalState,
    {
        fn minimize<S>(
            &self,
            boundary_conditions: &impl Constrain<X::Layout>,
            _: &(impl Load<X, ()> + Conservative<X, ()>),
            solver: S,
        ) -> Result<X, OptimizationError>
        where
            S: SolverFor<Self, (), ()> + Optimization<(), (), S::Tangent, X>,
            Self: ProvidesTangent<X, S::Tangent>,
        {
            solver.minimize(
                |_| Ok(()),
                |_| Ok(()),
                |x| self.provide_tangent(x).map_err(String::from),
                self.0.clone(),
                boundary_conditions.constraint(),
                None,
            )
        }
    }
    fn found<X>(result: Result<X, OptimizationError>) -> String {
        match result {
            Err(OptimizationError::Intermediate(indices)) => indices,
            _ => panic!("expected the recorded indices"),
        }
    }
    fn solid() -> Dummy<NodalCoordinates<3>> {
        Dummy(NodalCoordinates::<3>::from([[0.0, 0.0, 0.0]]))
    }
    fn coupled() -> Dummy<NodalStatesSolidThermal>
    where
        NodalStatesSolidThermal: NodalState,
    {
        Dummy(Vec::new().into())
    }
    #[test]
    fn the_layout_of_the_conditions_is_that_of_the_state() {
        let set = NodeSet::new(vec![0, 2]);
        let fixed = super::BoundaryConditions::new().fix(Displacement::<1>, &set);
        assert_eq!(found(solid().root(&fixed, &(), Recorder)), "[1, 7]");
    }
    #[test]
    fn the_layout_of_coupled_conditions_is_that_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(found(coupled().root(&fixed, &(), Recorder)), "[3, 4, 7]");
    }
    #[test]
    fn prescribed_values_reach_the_solver() {
        let fixed = super::BoundaryConditions::new().prescribe_node(
            Displacement::<1>,
            2,
            crate::math::Quantity::<crate::units::Length>::new(0.25),
        );
        assert_eq!(
            found(solid().root(&fixed, &(), Recorder)),
            "[(7, Offset(0.25))]"
        );
        assert_eq!(
            found(solid().minimize(&fixed, &(), Recorder)),
            "[(7, Offset(0.25))]"
        );
    }
    #[test]
    fn minimizing_follows_the_layout_of_the_state() {
        let set = NodeSet::new(vec![0, 2]);
        let fixed = super::BoundaryConditions::new().fix(Displacement::<1>, &set);
        assert_eq!(found(solid().minimize(&fixed, &(), Recorder)), "[1, 7]");
    }
    #[test]
    fn minimizing_coupled_conditions_follows_the_layout_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(
            found(coupled().minimize(&fixed, &(), Recorder)),
            "[3, 4, 7]"
        );
    }
    #[test]
    fn conditions_of_the_layout_may_be_written_out() {
        let set = NodeSet::new(vec![1]);
        let fixed =
            super::BoundaryConditions::<super::Solid<3>>::new().fix(Displacement::<2>, &set);
        assert_eq!(found(solid().root(&fixed, &(), Recorder)), "[5]");
    }
}
