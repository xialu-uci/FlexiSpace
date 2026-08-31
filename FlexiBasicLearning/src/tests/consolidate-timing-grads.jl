using JLD2, DataFrames

expdir_base = "../FlexiSpaceLocal/exp/grad-comp-timing/08302026/fixed-gts"
tasks_file  = "./scripts/timing_grads_tasks.txt"  # <- adjust to actual path

# One line per SLURM array task, in the same order used by `awk "NR==$SLURM_ARRAY_TASK_ID"`.
# Fields per line: diffname fname sname dof_str np_str  (same order as ARGS in timing_grads_parallel.jl)
task_lines = readlines(tasks_file)

missing_combos = NamedTuple[]
found = 0

for (task_id, line) in enumerate(task_lines)
    isempty(strip(line)) && continue
    fields = split(strip(line))
    if length(fields) != 5
        @warn "Unexpected number of fields on line $task_id: $line"
        continue
    end
    diffname, fname, sname, dof_str, np_str = fields
    d  = parse(Int, dof_str)
    np = parse(Int, np_str)

    outfile = joinpath(expdir_base, diffname, "task_results",
                        "$(diffname)_$(fname)_$(sname)_$(d)dof_$(np)np.jld2")

    if isfile(outfile)
        global found
        found += 1
    else
        push!(missing_combos, (task_id = task_id, diffname = diffname, fname = fname,
                                sname = sname, dof = d, num_points = np,
                                expected_file = outfile))
    end
end

println("Found $found / $(length(task_lines)) expected task files.")

if isempty(missing_combos)
    println("Nothing missing.")
else
    println("\nMissing $(length(missing_combos)) task file(s):\n")
    for m in missing_combos
        println("  task_id=$(m.task_id)  diffname=$(m.diffname)  fname=$(m.fname)  " *
                "shape=$(m.sname)  dof=$(m.dof)  num_points=$(m.num_points)")
    end

    # Directly usable for resubmission: sbatch --array=$(ids) your_job_script.sh
    ids = join([m.task_id for m in missing_combos], ",")
    println("\nSLURM array indices to rerun:\n--array=$ids")

    missing_df = DataFrame(missing_combos)
    outpath = joinpath(expdir_base, "missing_task_combos.jld2")
    JLD2.save(outpath, "missing", missing_df)
    println("\nSaved details to $outpath")
end