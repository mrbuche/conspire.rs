use crate::{
    constitutive::{
        fluid::viscoplastic::ViscoplasticFlow,
        hybrid::ElasticMultiplicativeViscoplastic,
        solid::{elastic::AlmansiHamel, hyperelastic::NeoHookean},
    },
    fem::{
        Blocks, ElasticViscoplasticAndElastic, FirstOrderRoot, Model, NodalCoordinates,
        NodalReferenceCoordinates,
        block::{
            Block,
            element::linear::{Hexahedron, Tetrahedron},
            solid::elastic_viscoplastic::ViscoplasticStateVariablesHistory,
        },
        solid::{
            elastic::ElasticFiniteElementModel,
            elastic_viscoplastic::FirstOrderRoot as DaeFirstOrderRoot,
        },
    },
    geometry::mesh::{Connectivity, Mesh},
    math::{
        Matrix, Scalar, Tensor, Vector,
        integrate::BogackiShampine,
        optimize::{EqualityConstraint, NewtonRaphson},
        test::{TestError, assert_eq, assert_eq_within_tols},
    },
};

const D: usize = 14;

type Tet = Block<AlmansiHamel, Tetrahedron, 1, 3, 4, 4>;
type Hex = Block<AlmansiHamel, Hexahedron, 8, 3, 8, 8>;
type TetNeoHookean = Block<NeoHookean, Tetrahedron, 1, 3, 4, 4>;
type TetViscoplastic = Block<
    ElasticMultiplicativeViscoplastic<AlmansiHamel, ViscoplasticFlow, Scalar>,
    Tetrahedron,
    1,
    3,
    4,
    4,
>;

fn constitutive_model() -> AlmansiHamel {
    AlmansiHamel {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    }
}

fn connectivity() -> Vec<[usize; 4]> {
    vec![
        [13, 12, 8, 1],
        [10, 3, 0, 8],
        [11, 10, 8, 3],
        [12, 11, 8, 2],
        [11, 2, 3, 8],
        [12, 2, 8, 1],
        [13, 10, 5, 0],
        [13, 11, 10, 8],
        [10, 6, 9, 5],
        [12, 7, 4, 9],
        [12, 11, 7, 9],
        [11, 7, 9, 6],
        [13, 1, 8, 0],
        [13, 9, 4, 5],
        [13, 12, 1, 4],
        [11, 10, 6, 9],
        [11, 10, 3, 6],
        [12, 11, 2, 7],
        [13, 11, 9, 10],
        [13, 12, 4, 9],
        [13, 10, 0, 8],
        [13, 10, 9, 5],
        [13, 12, 11, 8],
        [13, 12, 9, 11],
    ]
}

fn coordinates() -> NodalReferenceCoordinates {
    NodalReferenceCoordinates::from([
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [0.5, 0.5, -0.5],
        [0.5, -0.5, -0.5],
        [-0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [0.0, 0.0, 0.5],
        [0.0, 0.0, -0.5],
        [0.0, -0.5, 0.0],
        [-0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.0, 0.0],
    ])
}

fn deformed_coordinates() -> NodalCoordinates {
    NodalCoordinates::from([
        [0.48419081, -0.52698494, 0.42026988],
        [0.43559430, 0.52696224, 0.54477963],
        [-0.56594965, 0.57076191, 0.51683869],
        [-0.56061746, -0.42795457, 0.55275658],
        [0.41878700, 0.53190268, -0.44744274],
        [0.47232357, -0.57252738, -0.42946606],
        [-0.45168197, -0.5102938, -0.57959825],
        [-0.41776733, 0.41581785, -0.45911886],
        [0.05946988, 0.03773822, 0.44149305],
        [-0.08478334, -0.09009810, -0.46105872],
        [-0.04039882, -0.58201398, 0.09346960],
        [-0.57820738, 0.08325131, 0.03614415],
        [-0.04145077, 0.56406301, 0.09988905],
        [0.52149656, -0.08553510, -0.03187069],
    ])
}

fn constraint() -> (Matrix, Vector) {
    let strain = 0.55;
    let mut a = Matrix::zero(13, 3 * D);
    a[0][0] = 1.0;
    a[1][3] = 1.0;
    a[2][12] = 1.0;
    a[3][15] = 1.0;
    a[4][39] = 1.0;
    a[5][6] = 1.0;
    a[6][9] = 1.0;
    a[7][18] = 1.0;
    a[8][21] = 1.0;
    a[9][33] = 1.0;
    a[10][19] = 1.0;
    a[11][20] = 1.0;
    a[12][23] = 1.0;
    let mut b = Vector::zero(a.len());
    b[0] = 0.5 + strain;
    b[1] = 0.5 + strain;
    b[2] = 0.5 + strain;
    b[3] = 0.5 + strain;
    b[4] = 0.5 + strain;
    b[5] = -0.5;
    b[6] = -0.5;
    b[7] = -0.5;
    b[8] = -0.5;
    b[9] = -0.5;
    b[10] = -0.5;
    b[11] = -0.5;
    b[12] = -0.5;
    (a, b)
}

fn single_block_model() -> Result<Model<Tet>, TestError> {
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(connectivity().into())],
        coordinates(),
    ));
    (mesh, constitutive_model())
        .try_into()
        .map_err(|error: String| TestError { message: error })
}

fn neo_hookean_model() -> NeoHookean {
    NeoHookean {
        bulk_modulus: 13.0,
        shear_modulus: 3.0,
    }
}

fn split_connectivities() -> (Vec<[usize; 4]>, Vec<[usize; 4]>) {
    let mut connectivity_1 = connectivity();
    let connectivity_2 = connectivity_1.split_off(12);
    (connectivity_1, connectivity_2)
}

fn heterogeneous_model() -> Result<Model<Blocks<Tet, TetNeoHookean>>, TestError> {
    let (connectivity_1, connectivity_2) = split_connectivities();
    let mesh = Mesh::from((
        vec![
            Connectivity::Tetrahedral(connectivity_1.into()),
            Connectivity::Tetrahedral(connectivity_2.into()),
        ],
        coordinates(),
    ));
    (mesh, (constitutive_model(), neo_hookean_model()))
        .try_into()
        .map_err(|error: String| TestError { message: error })
}

fn split_blocks_model() -> Result<Model<Blocks<Tet, Tet>>, TestError> {
    let (connectivity_1, connectivity_2) = split_connectivities();
    let mesh = Mesh::from((
        vec![
            Connectivity::Tetrahedral(connectivity_1.into()),
            Connectivity::Tetrahedral(connectivity_2.into()),
        ],
        coordinates(),
    ));
    (mesh, (constitutive_model(), constitutive_model()))
        .try_into()
        .map_err(|error: String| TestError { message: error })
}

#[test]
fn scaled_jacobians() -> Result<(), TestError> {
    single_block_model()?
        .blocks
        .minimum_scaled_jacobians(&coordinates())
        .iter()
        .for_each(|scaled_jacobian| assert!(scaled_jacobian > &0.0));
    Ok(())
}

#[test]
fn single_block_nodal_forces() -> Result<(), TestError> {
    let block = Tet::from((constitutive_model(), connectivity(), &coordinates()));
    let model = single_block_model()?;
    assert_eq(
        &ElasticFiniteElementModel::nodal_forces(&block, &deformed_coordinates())?,
        &ElasticFiniteElementModel::nodal_forces(&model, &deformed_coordinates()).map_err(
            |error| TestError {
                message: error.to_string(),
            },
        )?,
    )
}

#[test]
fn split_blocks_nodal_forces() -> Result<(), TestError> {
    let block = Tet::from((constitutive_model(), connectivity(), &coordinates()));
    let model = split_blocks_model()?;
    assert_eq_within_tols(
        &ElasticFiniteElementModel::nodal_forces(&block, &deformed_coordinates())?,
        &ElasticFiniteElementModel::nodal_forces(&model, &deformed_coordinates()).map_err(
            |error| TestError {
                message: error.to_string(),
            },
        )?,
    )
}

#[test]
fn split_blocks_root() -> Result<(), TestError> {
    let (a, b) = constraint();
    let solution = FirstOrderRoot::root(
        &single_block_model()?,
        EqualityConstraint::Linear(a, b),
        NewtonRaphson::default(),
    )?;
    let (a, b) = constraint();
    let solution_split = FirstOrderRoot::root(
        &split_blocks_model()?,
        EqualityConstraint::Linear(a, b),
        NewtonRaphson::default(),
    )?;
    assert_eq_within_tols(&solution, &solution_split)
}

#[test]
fn wrong_block_count() {
    let (connectivity_1, connectivity_2) = split_connectivities();
    let mesh = Mesh::from((
        vec![
            Connectivity::Tetrahedral(connectivity_1.into()),
            Connectivity::Tetrahedral(connectivity_2.into()),
        ],
        coordinates(),
    ));
    let model: Result<Model<Tet>, String> = (mesh, constitutive_model()).try_into();
    assert!(model.unwrap_err().contains("expects 1"))
}

#[test]
fn wrong_element_kind() {
    let mesh = Mesh::from((
        vec![Connectivity::Tetrahedral(connectivity().into())],
        coordinates(),
    ));
    let model: Result<Model<Hex>, String> = (mesh, constitutive_model()).try_into();
    assert!(model.unwrap_err().contains("not hexahedral"))
}

#[test]
fn heterogeneous_blocks_nodal_forces() -> Result<(), TestError> {
    let (connectivity_1, connectivity_2) = split_connectivities();
    let block_1 = Tet::from((constitutive_model(), connectivity_1, &coordinates()));
    let block_2 = TetNeoHookean::from((neo_hookean_model(), connectivity_2, &coordinates()));
    assert_eq(
        &(ElasticFiniteElementModel::nodal_forces(&block_1, &deformed_coordinates())?
            + ElasticFiniteElementModel::nodal_forces(&block_2, &deformed_coordinates())?),
        &ElasticFiniteElementModel::nodal_forces(&heterogeneous_model()?, &deformed_coordinates())
            .map_err(|error| TestError {
                message: error.to_string(),
            })?,
    )
}

fn viscoplastic_model() -> ElasticMultiplicativeViscoplastic<AlmansiHamel, ViscoplasticFlow, Scalar>
{
    ElasticMultiplicativeViscoplastic::from((
        constitutive_model(),
        ViscoplasticFlow {
            yield_stress: 1e12,
            hardening_slope: 1.0,
            rate_sensitivity: 0.25,
            reference_flow_rate: 0.1,
        },
    ))
}

fn bcs(time: Scalar) -> EqualityConstraint {
    let (a, mut b) = constraint();
    (0..5).for_each(|i| b[i] = 0.5 + 0.55 * time);
    EqualityConstraint::Linear(a, b)
}

#[test]
fn mixed_viscoplastic_elastic_root() -> Result<(), TestError> {
    let (connectivity_1, connectivity_2) = split_connectivities();
    let mesh = Mesh::from((
        vec![
            Connectivity::Tetrahedral(connectivity_1.into()),
            Connectivity::Tetrahedral(connectivity_2.into()),
        ],
        coordinates(),
    ));
    let model: Model<ElasticViscoplasticAndElastic<TetViscoplastic, Tet>> =
        (mesh, (viscoplastic_model(), constitutive_model()))
            .try_into()
            .map_err(|error: String| TestError { message: error })?;
    let (_, coordinates_history, _): (_, _, ViscoplasticStateVariablesHistory<1, Scalar>) =
        DaeFirstOrderRoot::root(
            &model,
            BogackiShampine {
                abs_tol: 1e-6,
                rel_tol: 1e-6,
                ..Default::default()
            },
            NewtonRaphson::default(),
            &[0.0, 1.0],
            bcs,
        )?;
    let (a, b) = constraint();
    let reference = FirstOrderRoot::root(
        &split_blocks_model()?,
        EqualityConstraint::Linear(a, b),
        NewtonRaphson::default(),
    )?;
    assert_eq_within_tols(coordinates_history.iter().last().unwrap(), &reference)
}

#[test]
fn heterogeneous_blocks_root() -> Result<(), TestError> {
    let (a, b) = constraint();
    let model = heterogeneous_model()?;
    let solution = FirstOrderRoot::root(
        &model,
        EqualityConstraint::Linear(a, b),
        NewtonRaphson::default(),
    )?;
    assert!((solution[0][0] - 1.05).abs() < 1e-10);
    assert!((solution[2][0] + 0.5).abs() < 1e-10);
    let residual =
        ElasticFiniteElementModel::nodal_forces(&model, &solution).map_err(|error| TestError {
            message: error.to_string(),
        })?;
    [8, 9, 10, 12]
        .iter()
        .for_each(|&free_node| assert!(residual[free_node].norm() < 1e-10));
    Ok(())
}
