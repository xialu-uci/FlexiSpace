using FlexiBasicLearning
using JLD2, DataFrames

expdir_base = joinpath(FlexiBasicLearning.asset_dir, "exp/10012026/fixed-gts")
tasks_file  = "./scripts/timing_grads_tasks.txt"
# One line per SLURM array task, in the same order used by `awk "NR==$SLURM_ARRAY_TASK_ID"`.
# Fields per line: diffname fname sname dof_str np_str  (same order as ARGS in timing_grads_parallel.jl)
task_lines = readlines(tasks_file)

missing_combos = NamedTuple[]
rows_by_diff   = Dict{String, Vector{Any}}()   # diffname => rows loaded from task files
found = 0

for (task_id, line) in enumerate(task_lines)
    isempty(strip(line)) && continue
    fields = split(strip(line)) # strip gets rid of any trailing/leading whitespace, split produces a vector of strings by delimiter (whitespace by default)

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
        global found += 1 # find available task file
        row = JLD2.load(outfile, "row")
        # add diffname (not stored in the row by the task script) and task_id for traceability
        row = merge((diffname = String(diffname), task_id = task_id), row) 
        push!(get!(rows_by_diff, String(diffname), Any[]), row) # check diffname and add to rows_by_diff[diffname] 
    else
        push!(missing_combos, (task_id = task_id, diffname = diffname, fname = fname,
                               sname = sname, dof = d, num_points = np,
                               expected_file = outfile)) # find missing task file
    end
end

println("Found $found / $(length(task_lines)) expected task files.")

# build + save one DataFrame per diffname
for (diffname, rows) in rows_by_diff
    # DataFrame(::Vector{NamedTuple}) treats each NamedTuple as one row, so the
    # vector-valued grad_time_history field stays as a single cell per row.
    df = DataFrame(identity.(rows)) # identity used for type-narrowing, otherwise DataFrame constructor may promote to Any
    sort!(df, [:func_form, :shape, :dof, :num_points])

    outpath = joinpath(expdir_base, diffname, "$(diffname)_grad_time_results.jld2")
    JLD2.save(outpath, "df", df)
    println("Saved $(nrow(df))-row DataFrame for '$diffname' to $outpath")
end

# report missing
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