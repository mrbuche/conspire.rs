pub const DOC: &str = include_str!("doc.md");

pub fn fick<'a>() -> Vec<[&'a str; 2]> {
    vec![
        [
            "constitutive/chemical/diffusion/fick",
            include_str!("fick/doc.md"),
        ],
        ["potential", include_str!("fick/potential.md")],
        ["molar_flux", include_str!("fick/molar_flux.md")],
        [
            "molar_flux_tangent",
            include_str!("fick/molar_flux_tangent.md"),
        ],
    ]
}
