use crate::{
    domain::{
        NodalCoordinates,
        boundary::{
            BoundaryConditions, Concentration, Coupled, Direct, Displacement, Field, Layout, Left,
            NodalState, Potential, Right, Solid, SolidThermal, Temperature,
        },
        multiphysics::solid_thermal::NodalStatesSolidThermal,
    },
    geometry::mesh::{NodeSet, NodeSets},
    math::{
        Current, Quantity, QuantityVector, TensorRank1, TensorTuple, TensorTupleVec,
        optimize::EqualityConstraint,
    },
    units::{
        Concentration as ConcentrationUnit, ElectricPotential, Length,
        Temperature as TemperatureUnit,
    },
};
use std::marker::PhantomData;

fn indices(constraint: EqualityConstraint) -> Vec<usize> {
    match constraint {
        EqualityConstraint::Fixed(indices) => indices,
        _ => panic!("expected a fixed constraint"),
    }
}

fn sets() -> NodeSets {
    vec![
        NodeSet::new(vec![0, 2]).with_label(7).with_name("bottom"),
        NodeSet::new(vec![1]).with_name("top"),
    ]
    .into()
}

fn layout_of<X>() -> PhantomData<X::Layout>
where
    X: NodalState,
{
    PhantomData
}

#[test]
fn a_component_of_a_set() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix(Displacement::<1>, sets.by_name("bottom").unwrap());
    assert_eq!(indices(fixed.constraint()), [1, 7]);
}

#[test]
fn a_set_found_by_its_label() {
    let sets = sets();
    let fixed =
        BoundaryConditions::<Solid<3>>::new().fix(Displacement::<1>, sets.by_label(7).unwrap());
    assert_eq!(indices(fixed.constraint()), [1, 7]);
}

#[test]
fn a_solid_in_two_dimensions() {
    let sets = sets();
    let fixed =
        BoundaryConditions::<Solid<2>>::new().fix(Displacement::<1>, sets.by_name("top").unwrap());
    assert_eq!(indices(fixed.constraint()), [3]);
}

#[test]
fn every_component_of_a_node() {
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix_node(Displacement::<0>, 1)
        .fix_node(Displacement::<1>, 1)
        .fix_node(Displacement::<2>, 1);
    assert_eq!(indices(fixed.constraint()), [3, 4, 5]);
}

#[test]
fn temperatures_come_after_the_coordinates() {
    let fixed = BoundaryConditions::<SolidThermal<3>>::new().fix_nodes(Temperature, 0..3);
    assert_eq!(indices(fixed.constraint()), [3, 7, 11]);
}

#[test]
fn a_solid_and_its_temperature() {
    let fixed = BoundaryConditions::<SolidThermal<3>>::new()
        .fix_nodes(Temperature, 0..5)
        .fix_node(Displacement::<0>, 0)
        .fix_node(Displacement::<1>, 0)
        .fix_node(Displacement::<2>, 0)
        .fix_node(Displacement::<1>, 1)
        .fix_node(Displacement::<2>, 1)
        .fix_node(Displacement::<2>, 2);
    let mut expected: Vec<usize> = (0..5)
        .map(|node| 4 * node + 3)
        .chain([0, 1, 2, 5, 6, 10])
        .collect();
    expected.sort_unstable();
    assert_eq!(indices(fixed.constraint()), expected);
}

#[test]
fn a_scalar_field_is_indexed_by_its_node() {
    let sets = sets();
    let bottom = sets.by_name("bottom").unwrap();
    let temperature = BoundaryConditions::<Field<TemperatureUnit>>::new().fix(Temperature, bottom);
    let potential = BoundaryConditions::<Field<ElectricPotential>>::new().fix(Potential, bottom);
    let concentration =
        BoundaryConditions::<Field<ConcentrationUnit>>::new().fix(Concentration, bottom);
    assert_eq!(indices(temperature.constraint()), [0, 2]);
    assert_eq!(indices(potential.constraint()), [0, 2]);
    assert_eq!(indices(concentration.constraint()), [0, 2]);
}

#[test]
fn repeats_are_held_once() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix_nodes(Displacement::<0>, [1, 1, 2])
        .fix_node(Displacement::<0>, 1)
        .fix(Displacement::<0>, sets.by_name("bottom").unwrap());
    assert_eq!(indices(fixed.constraint()), [0, 3, 6]);
}

#[test]
fn nothing_fixed() {
    let fixed = BoundaryConditions::<Solid<3>>::new();
    assert!(indices(fixed.constraint()).is_empty());
}

#[test]
fn the_conditions_are_kept_in_order() {
    let sets = sets();
    let fixed = BoundaryConditions::<Solid<3>>::new()
        .fix(Displacement::<1>, sets.by_name("top").unwrap())
        .fix_node(Displacement::<2>, 4);
    let conditions = fixed.conditions();
    assert_eq!(conditions.len(), 2);
    assert_eq!(conditions[0].indices(), [4]);
    assert_eq!(conditions[1].indices(), [14]);
}

#[test]
fn a_condition_follows_the_set_it_borrows() {
    let sets = sets();
    let bottom = sets.by_name("bottom").unwrap();
    let before = BoundaryConditions::<Solid<3>>::new().fix(Displacement::<0>, bottom);
    let other = NodeSet::new(vec![5]);
    let after = BoundaryConditions::<Solid<3>>::new().fix(Displacement::<0>, &other);
    assert_eq!(indices(before.constraint()), [0, 6]);
    assert_eq!(indices(after.constraint()), [15]);
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
            boundary::{Displacement, NodalState, Temperature},
            multiphysics::solid_thermal::NodalStatesSolidThermal,
        },
        geometry::mesh::NodeSet,
        math::{
            optimize::{EqualityConstraint, Optimization, OptimizationError, RootFinding},
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
            equality_constraint: EqualityConstraint,
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
            equality_constraint: EqualityConstraint,
            _: Option<SparseSolver>,
        ) -> Result<X, OptimizationError> {
            record(equality_constraint)
        }
    }
    fn record<X>(equality_constraint: EqualityConstraint) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                Err(OptimizationError::Intermediate(format!("{indices:?}")))
            }
            _ => panic!("expected a fixed constraint"),
        }
    }
    impl<X> ProvidesTangent<X, ()> for Dummy<X> {
        fn provide_tangent(&self, _: &X) -> Result<(), ElementModelError> {
            Ok(())
        }
    }
    impl<X> Root<(), (), X> for Dummy<X>
    where
        X: Clone,
    {
        fn root<S>(
            &self,
            equality_constraint: EqualityConstraint,
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
                equality_constraint,
                None,
            )
        }
    }
    impl<X> Minimize<(), (), X> for Dummy<X>
    where
        X: Clone,
    {
        fn minimize<S>(
            &self,
            equality_constraint: EqualityConstraint,
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
                equality_constraint,
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
        assert_eq!(found(solid().root_with(&fixed, Recorder)), "[1, 7]");
    }
    #[test]
    fn the_layout_of_coupled_conditions_is_that_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(found(coupled().root_with(&fixed, Recorder)), "[3, 4, 7]");
    }
    #[test]
    fn minimizing_follows_the_layout_of_the_state() {
        let set = NodeSet::new(vec![0, 2]);
        let fixed = super::BoundaryConditions::new().fix(Displacement::<1>, &set);
        assert_eq!(found(solid().minimize_with(&fixed, Recorder)), "[1, 7]");
    }

    #[test]
    fn minimizing_coupled_conditions_follows_the_layout_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(
            found(coupled().minimize_with(&fixed, Recorder)),
            "[3, 4, 7]"
        );
    }

    #[test]
    fn conditions_of_the_layout_may_be_written_out() {
        let set = NodeSet::new(vec![1]);
        let fixed =
            super::BoundaryConditions::<super::Solid<3>>::new().fix(Displacement::<2>, &set);
        assert_eq!(found(solid().root_with(&fixed, Recorder)), "[5]");
    }
}
