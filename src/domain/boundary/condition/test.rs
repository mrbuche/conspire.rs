use crate::{
    domain::{
        NodalCoordinates,
        boundary::{
            BoundaryConditions, Concentration, Coupled, Direct, Displacement, Enforcement, Field,
            Layout, Left, NodalState, Potential, Right, Solid, SolidThermal, Temperature,
        },
        multiphysics::solid_thermal::NodalStatesSolidThermal,
    },
    geometry::mesh::{NodeSet, NodeSets},
    math::{
        Current, Quantity, QuantityVector, TensorRank1, TensorTuple, TensorTupleVec,
        optimize::{EqualityConstraint, EqualityConstraints},
    },
    units::{
        Concentration as ConcentrationUnit, ElectricPotential, Length,
        Temperature as TemperatureUnit,
    },
};
use std::marker::PhantomData;

fn indices(constraints: EqualityConstraints) -> Vec<usize> {
    EqualityConstraint::held_indices(&constraints).unwrap_or_default()
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

mod prescribed {
    use super::*;
    use crate::math::{Vector, optimize::Prescription};
    fn entries(constraints: EqualityConstraints) -> Vec<(usize, Prescription)> {
        constraints
            .into_iter()
            .flat_map(|constraint| match constraint {
                EqualityConstraint::Prescribed(entries) => entries,
                EqualityConstraint::Fixed(indices) => indices
                    .into_iter()
                    .map(|index| (index, Prescription::Initial))
                    .collect(),
                _ => panic!("expected a fixed or prescribed constraint"),
            })
            .collect()
    }
    #[test]
    fn a_displacement_is_held_at_a_distance_from_where_it_starts() {
        let sets = sets();
        let prescribed = BoundaryConditions::<Solid<3>>::new().prescribe(
            Displacement::<1>,
            sets.by_name("bottom").unwrap(),
            Quantity::<Length>::new(0.25),
        );
        assert_eq!(
            entries(prescribed.constraint()),
            [
                (1, Prescription::Offset(0.25)),
                (7, Prescription::Offset(0.25))
            ]
        );
    }
    #[test]
    fn a_field_is_held_at_its_value() {
        let prescribed = BoundaryConditions::<Field<TemperatureUnit>>::new().prescribe_nodes(
            Temperature,
            [3, 1],
            Quantity::<TemperatureUnit>::new(500.0),
        );
        assert_eq!(
            entries(prescribed.constraint()),
            [
                (3, Prescription::Absolute(500.0)),
                (1, Prescription::Absolute(500.0))
            ]
        );
    }
    #[test]
    fn every_field_is_held_at_its_value() {
        let prescribed = BoundaryConditions::<
            Coupled<Coupled<Solid<3>, Field<TemperatureUnit>>, Field<ElectricPotential>>,
        >::new()
        .prescribe_node(Temperature, 2, Quantity::<TemperatureUnit>::new(300.0))
        .prescribe_node(Potential, 2, Quantity::<ElectricPotential>::new(1.5));
        assert_eq!(
            entries(prescribed.constraint()),
            [
                (13, Prescription::Absolute(300.0)),
                (14, Prescription::Absolute(1.5))
            ]
        );
        let concentration = BoundaryConditions::<Field<ConcentrationUnit>>::new().prescribe_node(
            Concentration,
            4,
            Quantity::<ConcentrationUnit>::new(2.0),
        );
        assert_eq!(
            entries(concentration.constraint()),
            [(4, Prescription::Absolute(2.0))]
        );
    }
    #[test]
    fn a_velocity_is_held_at_its_value_in_its_own_layout() {
        use crate::{
            domain::{
                NodalVelocities,
                boundary::{SolidRate, Velocity},
            },
            units::Velocity as VelocityUnit,
        };
        let _: PhantomData<SolidRate<3>> = layout_of::<NodalVelocities<3>>();
        let rates = BoundaryConditions::<SolidRate<3>>::new().prescribe_node(
            Velocity::<1>,
            2,
            VelocityUnit::meters_per_second(0.5),
        );
        assert_eq!(
            entries(rates.constraint()),
            [(7, Prescription::Absolute(0.5))]
        );
        let held = BoundaryConditions::<SolidRate<2>>::new().fix_node(Velocity::<1>, 3);
        assert_eq!(indices(held.constraint()), [7]);
    }
    #[test]
    fn a_hold_among_values_is_held_at_its_initial_value() {
        let prescribed = BoundaryConditions::<Solid<3>>::new()
            .fix_node(Displacement::<0>, 0)
            .prescribe_node(Displacement::<2>, 0, Quantity::<Length>::new(0.5));
        assert_eq!(
            entries(prescribed.constraint()),
            [(0, Prescription::Initial), (2, Prescription::Offset(0.5))]
        );
    }
    #[test]
    fn only_holds_are_fixed() {
        let held = BoundaryConditions::<Solid<3>>::new().fix_node(Displacement::<0>, 1);
        assert_eq!(indices(held.constraint()), [3]);
    }
    #[test]
    fn the_conditions_know_what_they_hold() {
        let prescribed = BoundaryConditions::<Solid<3>>::new()
            .fix_node(Displacement::<0>, 0)
            .prescribe_node(Displacement::<1>, 0, Quantity::<Length>::new(0.5));
        let conditions = prescribed.conditions();
        assert_eq!(conditions[0].prescription(), Prescription::Initial);
        assert_eq!(conditions[1].prescription(), Prescription::Offset(0.5));
    }
    #[test]
    fn a_condition_is_eliminated_unless_made_otherwise() {
        let sets = sets();
        let bcs = BoundaryConditions::<Solid<3>>::new()
            .fix_node(Displacement::<0>, 0)
            .fix_by(
                Enforcement::Multipliers,
                Displacement::<1>,
                sets.by_name("bottom").unwrap(),
            )
            .prescribe_node(Displacement::<2>, 0, Quantity::<Length>::new(0.5))
            .prescribe_nodes_by(
                Enforcement::Multipliers,
                Displacement::<2>,
                [1, 2],
                Quantity::<Length>::new(0.25),
            )
            .prescribe_node_by(
                Enforcement::Elimination,
                Displacement::<0>,
                3,
                Quantity::<Length>::new(0.125),
            );
        let enforcements: Vec<Enforcement> = bcs
            .conditions()
            .iter()
            .map(|condition| condition.enforcement())
            .collect();
        assert_eq!(
            enforcements,
            [
                Enforcement::Elimination,
                Enforcement::Multipliers,
                Enforcement::Elimination,
                Enforcement::Multipliers,
                Enforcement::Elimination,
            ]
        );
    }
    #[test]
    fn a_condition_held_by_multipliers_keeps_its_indices_and_prescription() {
        let held = BoundaryConditions::<Solid<3>>::new()
            .fix_nodes_by(Enforcement::Multipliers, Displacement::<1>, [2, 0, 2])
            .prescribe_node_by(
                Enforcement::Multipliers,
                Displacement::<2>,
                1,
                Quantity::<Length>::new(0.5),
            )
            .prescribe_node(Displacement::<0>, 3, Quantity::<Length>::new(0.75))
            .fix_node(Displacement::<0>, 4);
        let constraints = held.constraint();
        assert_eq!(constraints.len(), 4);
        assert!(matches!(
            &constraints[0],
            EqualityConstraint::Multipliers(entries) if entries == &[
                (7, Prescription::Initial),
                (1, Prescription::Initial),
                (7, Prescription::Initial),
            ]
        ));
        assert!(matches!(
            &constraints[1],
            EqualityConstraint::Multipliers(entries) if entries == &[(5, Prescription::Offset(0.5))]
        ));
        assert!(matches!(
            &constraints[2],
            EqualityConstraint::Prescribed(entries) if entries == &[(9, Prescription::Offset(0.75))]
        ));
        assert!(matches!(
            &constraints[3],
            EqualityConstraint::Fixed(indices) if indices == &[12]
        ));
    }
    #[test]
    fn lowered_the_values_move_the_state() {
        let mut state = Vector::zero(4);
        let constraint = BoundaryConditions::<Solid<2>>::new()
            .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.5))
            .prescribe_node(Displacement::<0>, 0, Quantity::<Length>::new(-1.0))
            .constraint();
        let constraint = EqualityConstraint::lower(constraint, &mut state);
        assert_eq!(indices(constraint), [0, 3]);
        assert_eq!(state, Vector::from([-1.0, 0.0, 0.0, 0.5]));
    }
    #[test]
    fn a_later_condition_on_a_degree_of_freedom_wins() {
        let mut state = Vector::zero(4);
        let constraint = BoundaryConditions::<Solid<2>>::new()
            .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.5))
            .prescribe_node(Displacement::<1>, 1, Quantity::<Length>::new(0.75))
            .constraint();
        let constraint = EqualityConstraint::lower(constraint, &mut state);
        assert_eq!(indices(constraint), [3]);
        assert_eq!(state, Vector::from([0.0, 0.0, 0.0, 0.75]));
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
            boundary_conditions: &super::BoundaryConditions<'_, X::Layout>,
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
            boundary_conditions: &super::BoundaryConditions<'_, X::Layout>,
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
        assert_eq!(found(solid().root(&fixed, Recorder)), "[1, 7]");
    }
    #[test]
    fn the_layout_of_coupled_conditions_is_that_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(found(coupled().root(&fixed, Recorder)), "[3, 4, 7]");
    }
    #[test]
    fn prescribed_values_reach_the_solver() {
        let fixed = super::BoundaryConditions::new().prescribe_node(
            Displacement::<1>,
            2,
            crate::math::Quantity::<crate::units::Length>::new(0.25),
        );
        assert_eq!(found(solid().root(&fixed, Recorder)), "[(7, Offset(0.25))]");
        assert_eq!(
            found(solid().minimize(&fixed, Recorder)),
            "[(7, Offset(0.25))]"
        );
    }
    #[test]
    fn minimizing_follows_the_layout_of_the_state() {
        let set = NodeSet::new(vec![0, 2]);
        let fixed = super::BoundaryConditions::new().fix(Displacement::<1>, &set);
        assert_eq!(found(solid().minimize(&fixed, Recorder)), "[1, 7]");
    }
    #[test]
    fn minimizing_coupled_conditions_follows_the_layout_of_the_state() {
        let fixed = super::BoundaryConditions::new()
            .fix_nodes(Temperature, 0..2)
            .fix_node(Displacement::<0>, 1);
        assert_eq!(found(coupled().minimize(&fixed, Recorder)), "[3, 4, 7]");
    }
    #[test]
    fn conditions_of_the_layout_may_be_written_out() {
        let set = NodeSet::new(vec![1]);
        let fixed =
            super::BoundaryConditions::<super::Solid<3>>::new().fix(Displacement::<2>, &set);
        assert_eq!(found(solid().root(&fixed, Recorder)), "[5]");
    }
}
