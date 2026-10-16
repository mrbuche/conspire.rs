pub const DOC: &str = include_str!("doc.md");

pub fn ohmic<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/electrical/conduction/ohmic",
            include_str!("ohmic/doc.md"),
        ],
        ["potential", include_str!("ohmic/potential.md")],
        ["current_density", include_str!("ohmic/current_density.md")],
        [
            "current_density_tangent",
            include_str!("ohmic/current_density_tangent.md"),
        ],
    ]
}
