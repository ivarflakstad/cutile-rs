/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#[allow(dead_code)]
type Rank1 = cutile_dsl::core::Tile_1<f32, 8>;
#[allow(dead_code)]
type Rank2 = cutile_dsl::core::Tile_2<f32, 8, 8>;

#[allow(dead_code)]
type Fp8Tile = cutile_dsl::core::Tile_1<cutile_dsl::core::f8e4m3fn, 32>;

#[test]
fn the_module_ast_is_reachable() {
    let ast = cutile_dsl::core::__module_ast_self();
    assert_eq!(ast.name(), "core");
    assert_eq!(ast.absolute_path(), "cutile_dsl::_core::core");
}

#[test]
fn the_raw_tile_ir_surface_is_reachable_too() {
    let ast = cutile_dsl::tileir::__module_ast_self();
    assert_eq!(ast.name(), "tileir");
}
