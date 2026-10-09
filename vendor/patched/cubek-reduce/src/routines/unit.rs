use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    IdleMode, LineMode, ReduceError,
    launch::calculate_plane_count_per_cube,
    routines::{BlueprintStrategy, Routine, UnitReduceBlueprint, cube_count_safe},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient};

#[derive(Debug, Clone)]
pub struct UnitRoutine;

#[derive(Debug, Clone)]
pub struct UnitStrategy;

impl Routine for UnitRoutine {
    type Strategy = UnitStrategy;
    type Blueprint = UnitReduceBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, cube_count) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                let working_units = working_units(&settings, &problem);
                let num_units_in_cube = cube_dim.num_elems();
                let working_cubes = working_units.div_ceil(num_units_in_cube as usize);

                let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

                if working_cubes != launched_cubes && blueprint.unit_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many units launched for the problem causing OOD, but `unit_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Unit(blueprint),
                };

                (blueprint, cube_dim, cube_count)
            }
            BlueprintStrategy::Inferred(_) => {
                let (blueprint, cube_dim, cube_count) =
                    generate_blueprint::<R>(client, problem, &settings)?;
                (blueprint, cube_dim, cube_count)
            }
        };

        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceLineSettings,
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_units = working_units(settings, &problem);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, properties.num_cpu_cores);

    let plane_count =
        clamp_plane_count(plane_size, plane_count, properties.max_units_per_cube);
    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let num_units_in_cube = cube_dim.num_elems();

    let working_cubes = working_units.div_ceil(num_units_in_cube as usize);
    let (cube_count, cube_launched) = cube_count_safe(client, working_cubes);
    let unit_idle =
        !working_units.is_multiple_of(num_units_in_cube as usize) || cube_launched != working_cubes;

    let unit_idle = match unit_idle {
        true => IdleMode::Terminate,
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::Unit(UnitReduceBlueprint { unit_idle }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_units(settings: &ReduceLineSettings, problem: &ReduceProblem) -> usize {
    match settings.line_mode {
        LineMode::Parallel => problem.vector_count / settings.line_size_output,
        LineMode::Perpendicular => problem.vector_count / settings.line_size_input,
    }
}

/// Clamp a cube's plane count so the workgroup never exceeds the device's
/// `max_units_per_cube`.
///
/// `CubeDim::new` in cubecl-runtime applies this clamp itself, with the
/// comment "Make sure it respects the max units per cube (especially on
/// wasm)". The reduce routines build their `CubeDim` by hand and miss it.
///
/// On WebGPU that is not hypothetical. `cubecl-wgpu` cannot read a real
/// subgroup size from the WebGPU API, so it substitutes
/// `plane_size_max = 128`; with `plane_count = 4` the routines then ask for
/// a 512-invocation workgroup. WebGPU's *baseline*
/// `maxComputeInvocationsPerWorkgroup` is 256, so every adapter at the
/// baseline rejects the pipeline:
///
/// ```text
/// The total number of workgroup invocations (512) exceeds the maximum allowed (256).
/// [Invalid ComputePipeline "reduce_kernel"] is invalid due to a previous error.
/// ```
///
/// Discrete GPUs report 1024 and are unaffected, which is why this only
/// shows up on software adapters and low-end hardware.
///
/// Only the plane *count* (`cube_dim.y`) is reduced: `plane.rs` and
/// `cube.rs` both validate that `cube_dim.x == plane_size_max`, so `x` must
/// stay as it is.
fn clamp_plane_count(plane_size: u32, plane_count: u32, max_units_per_cube: u32) -> u32 {
    if max_units_per_cube == 0 {
        return plane_count;
    }
    let limit = (max_units_per_cube / plane_size.max(1)).max(1);
    plane_count.min(limit)
}
