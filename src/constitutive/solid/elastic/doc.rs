pub const DOC: &str = include_str!("doc.md");

pub fn almansi_hamel_eulerian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/almansi_hamel_eulerian",
            include_str!("almansi_hamel_eulerian/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("almansi_hamel_eulerian/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("almansi_hamel_eulerian/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
    ]
}

pub fn almansi_hamel_lagrangian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/almansi_hamel_lagrangian",
            include_str!("almansi_hamel_lagrangian/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("almansi_hamel_lagrangian/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("almansi_hamel_lagrangian/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
    ]
}

pub fn saint_venant_kirchhoff<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/saint_venant_kirchhoff",
            include_str!("saint_venant_kirchhoff/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("saint_venant_kirchhoff/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("saint_venant_kirchhoff/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
    ]
}

pub fn hencky<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/hencky",
            include_str!("hencky/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("hencky/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("hencky/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
    ]
}

pub fn bazant_itskov_eulerian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/bazant_itskov_eulerian",
            include_str!("bazant_itskov_eulerian/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("bazant_itskov_eulerian/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("bazant_itskov_eulerian/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
    ]
}

pub fn bazant_itskov_lagrangian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/bazant_itskov_lagrangian",
            include_str!("bazant_itskov_lagrangian/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("bazant_itskov_lagrangian/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("bazant_itskov_lagrangian/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
    ]
}

pub fn seth_hill_eulerian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/seth_hill_eulerian",
            include_str!("seth_hill_eulerian/doc.md"),
        ],
        [
            "cauchy_stress",
            include_str!("seth_hill_eulerian/cauchy_stress.md"),
        ],
        [
            "cauchy_tangent_stiffness",
            include_str!("seth_hill_eulerian/cauchy_tangent_stiffness.md"),
        ],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        ["second_piola_kirchhoff_stress", ""],
        ["second_piola_kirchhoff_tangent_stiffness", ""],
    ]
}

pub fn seth_hill_lagrangian<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/solid/elastic/seth_hill_lagrangian",
            include_str!("seth_hill_lagrangian/doc.md"),
        ],
        ["cauchy_stress", ""],
        ["cauchy_tangent_stiffness", ""],
        ["first_piola_kirchhoff_stress", ""],
        ["first_piola_kirchhoff_tangent_stiffness", ""],
        [
            "second_piola_kirchhoff_stress",
            include_str!("seth_hill_lagrangian/second_piola_kirchhoff_stress.md"),
        ],
        [
            "second_piola_kirchhoff_tangent_stiffness",
            include_str!("seth_hill_lagrangian/second_piola_kirchhoff_tangent_stiffness.md"),
        ],
    ]
}
