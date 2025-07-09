pub const DOC: &str = include_str!("doc.md");

pub fn arruda_boyce<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/arruda_boyce",
            include_str!("arruda_boyce/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("arruda_boyce/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("arruda_boyce/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("arruda_boyce/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn fung<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/fung",
            include_str!("fung/doc.md"),
        ],
        ["cauchy_stress", include_str!("fung/cauchy_stress.md")],
        [
            "cauchy_tangent_stiffness",
            include_str!("fung/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("fung/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn gent<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/gent",
            include_str!("gent/doc.md"),
        ],
        ["cauchy_stress", include_str!("gent/cauchy_stress.md")],
        [
            "cauchy_tangent_stiffness",
            include_str!("gent/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("gent/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn mooney_rivlin<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/mooney_rivlin",
            include_str!("mooney_rivlin/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("mooney_rivlin/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("mooney_rivlin/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("mooney_rivlin/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn neo_hookean<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/neo_hookean",
            include_str!("neo_hookean/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("neo_hookean/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("neo_hookean/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("neo_hookean/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn saint_venant_kirchhoff<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/saint_venant_kirchhoff",
            include_str!("saint_venant_kirchhoff/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("saint_venant_kirchhoff/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
        [
            "helmholtz_free_energy_density",
            include_str!("saint_venant_kirchhoff/helmholtz_free_energy_density.md"),
        ],
    ]
}

pub fn yeoh<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/hyperelastic/yeoh",
            include_str!("yeoh/doc.md"),
        ],
        ["cauchy_stress", include_str!("yeoh/cauchy_stress.md")],
        [
            "cauchy_tangent_stiffness",
            include_str!("yeoh/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
        [
            "helmholtz_free_energy_density",
            include_str!("yeoh/helmholtz_free_energy_density.md"),
        ],
    ]
}
