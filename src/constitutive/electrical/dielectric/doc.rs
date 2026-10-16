pub const DOC: &str = include_str!("doc.md");

pub fn linear<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/electrical/dielectric/linear",
            include_str!("linear/doc.md"),
        ],
        ["potential", include_str!("linear/potential.md")],
        [
            "electric_displacement",
            include_str!("linear/electric_displacement.md"),
        ],
        [
            "electric_displacement_tangent",
            include_str!("linear/electric_displacement_tangent.md"),
        ],
    ]
}
