use crate::{
    constitutive::thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
    domain::ElementModel,
    fem::{
        Model,
        block::{Block, element::linear::Hexahedron},
        boundary::{ConservativeAt, Field, Loads},
        diffusion::{NodalFields, NodalForces},
    },
    geometry::mesh::{Connectivity, Mesh, SideSet},
    math::{Quantity, Tensor},
    units::{Power, PowerPerArea, Temperature, Time},
};

type Hex = Block<Fourier, Hexahedron, 8, 3, 8, 8>;

fn model() -> Model<Hex, 3> {
    let mut mesh = Mesh::from((
        vec![Connectivity::Hexahedral(
            vec![[0, 1, 2, 3, 4, 5, 6, 7]].into(),
        )],
        vec![
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 3.0, 1.0],
            [0.0, 3.0, 1.0],
        ]
        .into(),
    ));
    mesh.set_side_sets(vec![SideSet::new(vec![(0, 5)])].into());
    (
        mesh,
        Fourier {
            thermal_conductivity: THERMAL_CONDUCTIVITY,
        },
    )
        .try_into()
        .unwrap()
}

fn residual_at(
    loads: &impl crate::fem::boundary::LoadAt<NodalFields<Power>, NodalForces<Power>>,
    time: f64,
) -> Vec<f64> {
    let mut residual = NodalForces::<Power>::zero(8);
    loads
        .subtract_at(
            Time::seconds(time),
            &NodalFields::<Power>::zero(8),
            &mut residual,
        )
        .unwrap();
    residual.iter().map(|power| power.value()).collect()
}

#[test]
fn a_flux_through_a_face_gives_each_node_its_share_of_the_area() {
    let model = model();
    let loads = Loads::<Field<Temperature>>::new()
        .flux::<Power>(
            model.coordinates(),
            model.side_sets().iter().next().unwrap(),
            Quantity::<PowerPerArea>::new(5.0),
        )
        .unwrap();
    let residual = residual_at(&loads, 0.0);
    (0..4).for_each(|node| assert_eq!(residual[node], 0.0));
    (4..8).for_each(|node| assert!((residual[node] + 5.0 * 6.0 / 4.0).abs() < 1e-12));
}

#[test]
fn a_flux_that_depends_on_time_takes_the_time_it_is_given() {
    let model = model();
    let loads = Loads::<Field<Temperature>>::new()
        .flux_with::<Power>(
            model.coordinates(),
            model.side_sets().iter().next().unwrap(),
            |time| Quantity::<PowerPerArea>::new(time.value()),
        )
        .unwrap();
    (4..8).for_each(|node| {
        assert!((residual_at(&loads, 2.0)[node] + 2.0 * 6.0 / 4.0).abs() < 1e-12);
        assert!((residual_at(&loads, 3.0)[node] + 3.0 * 6.0 / 4.0).abs() < 1e-12)
    });
}

#[test]
fn a_flux_through_faces_the_mesh_does_not_have_is_an_error() {
    let model = model();
    let error = Loads::<Field<Temperature>>::new()
        .flux::<Power>(
            model.coordinates(),
            &SideSet::new(vec![(0, 5)]),
            Quantity::<PowerPerArea>::new(1.0),
        )
        .err()
        .unwrap();
    assert!(error.contains("no faces"));
}

#[test]
fn a_flux_through_the_top_of_a_held_bar_raises_it_by_the_flux_times_the_length_over_the_conductivity()
 {
    use crate::{
        fem::{
            Root,
            boundary::{BoundaryConditions, ConstrainAt as _},
            diffusion::NodalStiffnesses,
        },
        math::optimize::NewtonRaphson,
    };
    let model = model();
    let conditions = BoundaryConditions::<Field<Temperature>>::new()
        .fix_nodes(crate::fem::boundary::Temperature, 0..4);
    let loads = Loads::<Field<Temperature>>::new()
        .flux::<Power>(
            model.coordinates(),
            model.side_sets().iter().next().unwrap(),
            Quantity::<PowerPerArea>::new(5.0),
        )
        .unwrap();
    let temperatures = <Model<Hex, 3> as Root<
        NodalForces<Power>,
        NodalStiffnesses<Power>,
        NodalFields<Power>,
    >>::root(&model, &conditions, &loads, NewtonRaphson::default())
    .unwrap();
    let rise = 5.0 * 1.0 / THERMAL_CONDUCTIVITY.value();
    (0..4).for_each(|node| assert!(temperatures[node].value().abs() < 1e-9));
    (4..8).for_each(|node| {
        let found = temperatures[node].value();
        assert!(
            (found - rise).abs() < 1e-9 * rise.abs().max(1.0),
            "node {node}: {found} != {rise}"
        )
    });
}

fn temperatures() -> NodalFields<Power> {
    (0..8)
        .map(|node| Quantity::<Temperature>::new(node as f64))
        .collect()
}

#[test]
fn the_energy_of_a_flux_that_depends_on_time_is_that_of_the_flux_at_the_time() {
    let model = model();
    let side_set = model.side_sets().iter().next().unwrap();
    let loads = Loads::<Field<Temperature>>::new()
        .flux_with::<Power>(model.coordinates(), side_set, |time| {
            Quantity::<PowerPerArea>::new(time.value())
        })
        .unwrap();
    let (early, late) = (
        loads
            .potential_energy_at(Time::seconds(2.0), &temperatures())
            .unwrap(),
        loads
            .potential_energy_at(Time::seconds(3.0), &temperatures())
            .unwrap(),
    );
    let top = (4..8).sum::<usize>() as f64 * 6.0 / 4.0;
    assert!((early.value() + 2.0 * top).abs() < 1e-12);
    assert!((late.value() + 3.0 * top).abs() < 1e-12);
}

#[test]
fn a_flux_that_does_not_change_has_the_same_energy_whether_or_not_it_could() {
    let model = model();
    let side_set = model.side_sets().iter().next().unwrap();
    let timed = Loads::<Field<Temperature>>::new()
        .flux_with::<Power>(model.coordinates(), side_set, |_| {
            Quantity::<PowerPerArea>::new(5.0)
        })
        .unwrap();
    let fixed = Loads::<Field<Temperature>>::new()
        .flux::<Power>(
            model.coordinates(),
            side_set,
            Quantity::<PowerPerArea>::new(5.0),
        )
        .unwrap();
    let at = Time::seconds(7.0);
    assert!(
        (timed
            .potential_energy_at(at, &temperatures())
            .unwrap()
            .value()
            - fixed
                .potential_energy_at(at, &temperatures())
                .unwrap()
                .value())
        .abs()
            < 1e-12
    );
}

#[test]
fn the_products_of_a_flux_and_of_an_energy_exist_for_every_balance_law() {
    use crate::{
        constitutive::diffusion::Diffusive,
        units::{Area, Charge, Current, MolarFlowRate},
    };
    use std::ops::Mul;
    fn typed<L>()
    where
        L: Diffusive,
        Quantity<L::Flux>: Mul<Quantity<Area>, Output = Quantity<L>>,
        Quantity<L::Field>: Mul<Quantity<L>, Output = Quantity<L::Energy>>,
    {
    }
    typed::<Power>();
    typed::<MolarFlowRate>();
    typed::<Charge>();
    typed::<Current>();
}

#[test]
fn a_heat_flux_that_depends_on_time_leaves_the_forces_of_a_coupled_residual_alone() {
    use crate::{
        domain::multiphysics::solid_thermal::{
            NodalForcesSolidThermal, NodalStatesSolidThermal, join_states,
        },
        fem::{
            NodalCoordinates,
            boundary::{LoadAt, SolidThermal},
        },
    };
    let model = model();
    let loads = Loads::<SolidThermal<3>>::new()
        .flux_with(
            model.coordinates(),
            model.side_sets().iter().next().unwrap(),
            |time| Quantity::<PowerPerArea>::new(time.value()),
        )
        .unwrap();
    let coordinates: NodalCoordinates<3> = model.coordinates().clone().into();
    let states: NodalStatesSolidThermal = join_states(&coordinates, &NodalFields::<Power>::zero(8));
    let mut residual = (0..8)
        .map(|_| {
            crate::math::TensorTuple(
                crate::mechanics::Force::from([0.0; 3]),
                Quantity::<crate::units::Power>::new(0.0),
            )
        })
        .collect::<NodalForcesSolidThermal>();
    loads
        .subtract_at(Time::seconds(2.0), &states, &mut residual)
        .unwrap();
    (0..8).for_each(|node| {
        assert!(residual[node].0.iter().all(|force| force.value() == 0.0));
        let heat = if node < 4 { 0.0 } else { -2.0 * 6.0 / 4.0 };
        assert!((residual[node].1.value() - heat).abs() < 1e-12)
    });
}
