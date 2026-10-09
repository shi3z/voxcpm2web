use super::{
    GlobalReduceBlueprint, ReduceBlueprint, ReduceLaunchSettings, ReduceLineSettings, ReduceProblem,
};
use crate::{
    BoundChecks, IdleMode, LineMode, ReduceError,
    launch::{calculate_plane_count_per_cube, support_plane},
    routines::{BlueprintStrategy, CubeBlueprint, Routine, cube_count_safe},
};
use cubecl::{CubeCount, CubeDim, Runtime, client::ComputeClient, features::Plane};

#[derive(Debug, Clone)]
pub struct CubeRoutine;

#[derive(Debug, Clone)]
pub struct CubeStrategy {
    /// If we use plane to aggregate accumulators.
    pub use_planes: bool,
}

impl Routine for CubeRoutine {
    type Strategy = CubeStrategy;
    type Blueprint = CubeBlueprint;

    fn prepare<R: Runtime>(
        &self,
        client: &ComputeClient<R>,
        problem: ReduceProblem,
        settings: ReduceLineSettings,
        strategy: BlueprintStrategy<Self>,
    ) -> Result<(ReduceBlueprint, ReduceLaunchSettings), ReduceError> {
        let (blueprint, cube_dim, num_cubes) = match strategy {
            BlueprintStrategy::Forced(blueprint, cube_dim) => {
                // One accumulator per plane.
                if blueprint.use_planes {
                    if !support_plane(client) {
                        return Err(ReduceError::PlanesUnavailable);
                    }

                    if blueprint.num_shared_accumulators != cube_dim.x as usize {
                        return Err(ReduceError::Validation {
                            details: "Num accumulators should match cube_dim.x",
                        });
                    }
                    if cube_dim.x != client.properties().hardware.plane_size_max {
                        return Err(ReduceError::Validation {
                            details: "`cube_dim.x` must match `plane_size_max`",
                        });
                    }
                // One accumulator per unit.
                } else if blueprint.num_shared_accumulators != cube_dim.num_elems() as usize {
                    return Err(ReduceError::Validation {
                        details: "Num accumulators should match cube_dim.num_elems()",
                    });
                }

                let working_cubes = working_cubes(&settings, &problem);
                let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

                if working_cubes != launched_cubes && !blueprint.cube_idle.is_enabled() {
                    return Err(ReduceError::Validation {
                        details: "Too many cubes launched for the problem causing OOD, but `cube_idle` is off.",
                    });
                }

                let blueprint = ReduceBlueprint {
                    line_mode: settings.line_mode,
                    global: GlobalReduceBlueprint::Cube(blueprint),
                };

                (blueprint, cube_dim, cube_count)
            }
            BlueprintStrategy::Inferred(strategy) => {
                let (blueprint, cube_dim, cube_count) =
                    generate_blueprint::<R>(client, problem, &settings, strategy)?;
                (blueprint, cube_dim, cube_count)
            }
        };

        let launch = ReduceLaunchSettings {
            cube_dim,
            cube_count: num_cubes,
            line: settings,
        };

        Ok((blueprint, launch))
    }
}

fn generate_blueprint<R: Runtime>(
    client: &ComputeClient<R>,
    problem: ReduceProblem,
    settings: &ReduceLineSettings,
    strategy: CubeStrategy,
) -> Result<(ReduceBlueprint, CubeDim, CubeCount), ReduceError> {
    if strategy.use_planes && !support_plane(client) {
        return Err(ReduceError::PlanesUnavailable);
    }

    let properties = &client.properties().hardware;
    let plane_size = properties.plane_size_max;
    let working_cubes = working_cubes(settings, &problem);
    let working_units = working_cubes * problem.vector_size.div_ceil(settings.line_size_input);
    let plane_count =
        calculate_plane_count_per_cube(working_units, plane_size, properties.num_cpu_cores);
    let plane_count =
        clamp_plane_count(plane_size, plane_count, properties.max_units_per_cube);
    let cube_dim = CubeDim::new_2d(plane_size, plane_count);
    let cube_size = cube_dim.num_elems();

    let work_size = match settings.line_mode {
        LineMode::Parallel => problem.vector_size / settings.line_size_input,
        LineMode::Perpendicular => problem.vector_size,
    };
    let bound_checks = match work_size.is_multiple_of(cube_size as usize) {
        true => BoundChecks::None,
        false => BoundChecks::Mask,
    };

    let num_shared_accumulators = match strategy.use_planes {
        true => plane_count as usize,
        false => cube_size as usize,
    };

    let (cube_count, launched_cubes) = cube_count_safe(client, working_cubes);

    let cube_idle = match working_cubes != launched_cubes {
        true => match strategy.use_planes
            && !client
                .properties()
                .features
                .plane
                .contains(Plane::NonUniformControlFlow)
        {
            true => IdleMode::Mask,
            false => IdleMode::Terminate,
        },
        false => IdleMode::None,
    };
    let blueprint = ReduceBlueprint {
        line_mode: settings.line_mode,
        global: GlobalReduceBlueprint::Cube(CubeBlueprint {
            cube_idle,
            bound_checks,
            num_shared_accumulators,
            use_planes: strategy.use_planes,
        }),
    };

    Ok((blueprint, cube_dim, cube_count))
}

fn working_cubes(settings: &ReduceLineSettings, problem: &ReduceProblem) -> usize {
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
