# Plan: Conspire & Automesh Focus

Focus on finite element quality metrics, specifically the Minimum Scaled Jacobian (MSJ), and its integration between `conspire` and `automesh`.

## Phase 1: Coverage and Validation of MSJ in `conspire`

1.  **Audit MSJ Implementation:**
    *   Examine `minimum_scaled_jacobian` and `scaled_jacobians` for all linear hexahedral finite elements:
        *   `src/domain/fem/block/element/linear/hexahedron/mod.rs`
        *   `src/domain/fem/block/element/linear/tetrahedron/mod.rs`
    * Ignore these for now:
        *   `src/domain/fem/block/element/linear/wedge/mod.rs`
        *   `src/domain/fem/block/element/linear/pyramid/mod.rs`
        *   `src/domain/fem/block/element/surface/linear/triangle/mod.rs`
        *   `src/domain/fem/block/element/surface/linear/quadrilateral/mod.rs`
    *   Check for potential numerical stability issues (e.g., zero-length edges).
2.  **Develop Test Suite for MSJ:**
    *   Create a new test module or extend existing `test.rs` files for each element.
    *   Test cases should include:
        *   Ideal/Unit elements (MSJ should be 1.0 or close for equilateral/orthogonal elements).
        *   Degenerate/Flat elements (MSJ should be 0.0).
        *   Inverted elements (MSJ should be negative).
        *   Perturbed/Distorted elements with known MSJ values.
3.  **Global MSJ at Block Level:**
    *   Verify `minimum_scaled_jacobians` in `src/domain/fem/block/mod.rs` correctly aggregates element-wise MSJ.
    *   Add block-level tests for `Block::minimum_scaled_jacobians`.

## Phase 2: Automesh Integration Research

1.  **Analyze `automesh` Usage:**
    *   Investigate `automesh/src/fem/hex/mod.rs` to see how `conspire::fem::block::element::linear::Hexahedron` is utilized.
    *   Identify how `automesh` uses MSJ for mesh quality assessment or optimization.
2.  **Bridge Improvements:**
    *   Based on `automesh` needs, determine if additional metrics (e.g., Aspect Ratio, Skewness) or higher-order element support are needed in `conspire`.

## Phase 3: Implementation and Refactoring

1.  **Refactor for Performance/Clarity:**
    *   Optimize MSJ calculations if necessary (e.g., reusing vector calculations).
2.  **Sync with Automesh:**
    *   Ensure `automesh` can seamlessly leverage the updated `conspire` features.
