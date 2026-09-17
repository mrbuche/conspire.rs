use crate::{
    constitutive::solid::elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
    domain::{
        NodalCoordinates, NodalReferenceCoordinates, NodalVelocities,
        block::test::{
            test_finite_element_block_with_elastic_constitutive_model,
            test_finite_element_block_with_hyperelastic_constitutive_model,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::Tensor,
    mechanics::{
        DeformationGradient, test::get_deformation_gradient, test::get_deformation_gradient_rate,
    },
    vem::block::{Block, element::Element},
};

const D: usize = 21;

fn get_element_face_connectivity() -> Vec<Vec<usize>> {
    vec![
        vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        vec![12, 13, 14, 15, 16, 17],
    ]
}

fn get_face_node_connectivity() -> Vec<Vec<usize>> {
    vec![
        vec![16, 17, 4, 8, 0],
        vec![12, 13, 2, 16, 0],
        vec![8, 9, 1, 12, 0],
        vec![9, 5, 19, 18, 1],
        vec![18, 3, 13, 12, 1],
        vec![10, 6, 17, 16, 2],
        vec![13, 3, 11, 10, 2],
        vec![7, 11, 3, 18, 19],
        vec![14, 5, 9, 8, 4],
        vec![6, 15, 14, 4, 17],
        vec![5, 14, 15, 7, 19],
        vec![6, 10, 11, 7, 15],
        vec![0, 8, 4, 17, 16],
        vec![8, 0, 20],
        vec![4, 8, 20],
        vec![17, 4, 20],
        vec![16, 17, 20],
        vec![0, 16, 20],
    ]
}

fn get_reference_coordinates_block() -> NodalReferenceCoordinates<3> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    NodalReferenceCoordinates::from([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [0.0, -phi, -1.0 / phi],
        [0.0, -phi, 1.0 / phi],
        [0.0, phi, -1.0 / phi],
        [0.0, phi, 1.0 / phi],
        [-phi, -1.0 / phi, 0.0],
        [-phi, 1.0 / phi, 0.0],
        [phi, -1.0 / phi, 0.0],
        [phi, 1.0 / phi, 0.0],
        [-1.0 / phi, 0.0, -phi],
        [1.0 / phi, 0.0, -phi],
        [-1.0 / phi, 0.0, phi],
        [1.0 / phi, 0.0, phi],
        [0.0, -1.25, -2.0],
    ])
}

fn get_coordinates_block() -> NodalCoordinates<3> {
    get_reference_coordinates_block()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient() * reference_coordinate)
        .collect()
}

fn get_velocities_block() -> NodalVelocities<3> {
    get_reference_coordinates_block()
        .iter()
        .map(|reference_coordinate| get_deformation_gradient_rate() * reference_coordinate)
        .collect()
}

macro_rules! setup_block {
    ($constitutive_model: expr, $constitutive_model_type: ident) => {
        fn get_block() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_block(),
            ))
        }
        fn get_block_transformed() -> Block<$constitutive_model_type, Element> {
            Block::<$constitutive_model_type, Element>::from((
                $constitutive_model,
                get_element_face_connectivity(),
                get_face_node_connectivity(),
                &get_reference_coordinates_transformed_block(),
            ))
        }
    };
}

crate::domain::block::test::test_block_elastic_and_hyperelastic!(Element);

mod block_viscous {
    use super::*;
    use crate::{
        EPSILON,
        domain::block::test::{
            test_finite_element_block_with_elastic_hyperviscous_constitutive_model,
            test_finite_element_block_with_hyperviscoelastic_constitutive_model,
        },
        math::{Rank2, TensorRank2, assert::AssertionError},
        mechanics::test::{
            get_rotation_current_configuration, get_rotation_rate_current_configuration,
            get_rotation_reference_configuration, get_translation_current_configuration,
            get_translation_rate_current_configuration, get_translation_reference_configuration,
        },
    };
    mod elastic_hyperviscous {
        use super::*;
        use crate::{
            constitutive::{
                canonical::Canonical,
                fluid::hyperviscous::Newtonian,
                solid::{
                    elastic::AlmansiHamelEulerian,
                    elastic_hyperviscous::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                },
            },
            domain::solid::NodalDampingsSolid,
            domain::solid::{
                elastic_hyperviscous::ElasticHyperviscousElements,
                viscoelastic::ViscoelasticElements,
            },
        };
        type AlmansiHamel = Canonical<AlmansiHamelEulerian, Newtonian>;
        mod almansi_hamel {
            use super::*;
            test_finite_element_block_with_elastic_hyperviscous_constitutive_model!(
                Element,
                Element,
                AlmansiHamel::from((
                    AlmansiHamelEulerian {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    Newtonian {
                        bulk_viscosity: BULK_VISCOSITY,
                        shear_viscosity: SHEAR_VISCOSITY,
                    },
                )),
                AlmansiHamel
            );
        }
    }
    mod hyperviscoelastic {
        use super::*;
        use crate::{
            constitutive::{
                canonical::Canonical,
                fluid::hyperviscous::SaintVenantKirchhoff as ViscousSaintVenantKirchhoff,
                solid::{
                    hyperelastic::SaintVenantKirchhoff as HyperelasticSaintVenantKirchhoff,
                    hyperviscoelastic::test::{BULK_VISCOSITY, SHEAR_VISCOSITY},
                },
            },
            domain::solid::NodalDampingsSolid,
            domain::solid::{
                elastic_hyperviscous::ElasticHyperviscousElements,
                viscoelastic::ViscoelasticElements,
            },
        };
        type SaintVenantKirchhoff =
            Canonical<HyperelasticSaintVenantKirchhoff, ViscousSaintVenantKirchhoff>;
        mod saint_venant_kirchhoff {
            use super::*;
            test_finite_element_block_with_hyperviscoelastic_constitutive_model!(
                Element,
                Element,
                SaintVenantKirchhoff::from((
                    HyperelasticSaintVenantKirchhoff {
                        bulk_modulus: BULK_MODULUS,
                        shear_modulus: SHEAR_MODULUS,
                    },
                    ViscousSaintVenantKirchhoff {
                        bulk_viscosity: BULK_VISCOSITY,
                        shear_viscosity: SHEAR_VISCOSITY,
                    },
                )),
                SaintVenantKirchhoff
            );
        }
    }
}
