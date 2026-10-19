use crate::{
    constitutive::thermal::conduction::{Fourier, test::THERMAL_CONDUCTIVITY},
    domain::ElementModel,
    fem::{
        Model,
        block::{Block, element::linear::Hexahedron},
        boundary::{Field, Loads},
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
