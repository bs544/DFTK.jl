include("run_scf_and_compare.jl")
include("testcases.jl")

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [4.0,3.0]
    a = 5.42352
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    model = model_PBE(lattice, atoms, positions;
                      temperature=0.01, magnetic_moments)
    
    model = convert(Model{Float64}, model)
    
    

    idx = 1
    resid_weight = 0.05 # 0.5 works for simple mixing with α=0.2
    r_sm_frac    = 0.1
    r_cut        = 2.0
    α            = 0.8
    charge       = 4.736
    spin = 1.5

    # constraints = [DFTK.Constraint(model,idx,resid_weight,r_sm_frac;target_charge=charge,r_cut)]
    

    cons_infos = []
    energies = []
    n_steps = []
    V = nothing
    
    solver = scf_anderson_solver()

    energies    = []
    constraint_info = []
    n_steps     = []

    function add_constraint!(terms,constraint_term)
        constraint_term_idx = nothing

        for (i,term) in enumerate(terms)
            if typeof(term)==DFTK.DensityMixingConstraint
                constraint_term_idx = i
                break
            end
        end

        if isnothing(constraint_term_idx)
            push!(terms,constraint_term)
        else
            terms[constraint_term_idx] = constraint_term
        end
    end

    # detail | resid_weight | # steps
    # None   | 0.5          | 31
    # approx | 0.5          | 51
    # nonint | 38,39,42|41,40 => no difference
    # for detail ∈ ["None","approximate","noninteracting"]
        # resid_weight=0.5
    for resid_weight in [0.7,0.7]
        detail = "noninteracting"
        spin = 1.5
        constraints = [DFTK.Constraint(model,idx,resid_weight,r_sm_frac;target_spin=spin,r_cut),DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin+0.1,r_cut)]
        constraint_term = DFTK.DensityMixingConstraint(constraints)
        terms = model.term_types
        add_constraint!(terms,constraint_term)
        model = Model(model;terms)
        initial_lambdas = nothing #[0.0 0.008; 0.0 0.0078]
        basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
        # resid_weight *= basis.dvol
        # constraints = DFTK.get_constraints(basis)
        # c1_arr = constraints.at_fn_arrays[1,2]
        # c2_arr = constraints.at_fn_arrays[2,2]
        # println(c1_arr[1,1,:])
        # println(c2_arr[1,1,:])
        ρ0 = guess_density(basis,magnetic_moments)
        scfres = DFTK.density_mixed_constrained(basis; solver=solver, tol=1.0e-10,ρ=ρ0,maxiter=1000,damping=α,lambdas_preconditioning=detail,initial_lambdas)
        DFTK.display_constraints(scfres.constraints)
        push!(energies,scfres.energies.total)
        push!(constraint_info,scfres.constraints)
        push!(n_steps,scfres.n_iter)
    end
    
    println("The zero for λ should correspond to the energy minimum")
    for (i,constraint) in enumerate(constraint_info)
        λ_string  = string(constraint.lambdas[1,2])
        e_string  = string(energies[i]-energies[1])
        st_string = string(constraint.target_values[1,2])
        sv_string = string(constraint.current_values[1,2])
        
        λ_string  = λ_string[begin:min(length(λ_string),7)]
        e_string  = e_string[begin:min(length(e_string),7)]
        st_string = st_string[begin:min(length(st_string),5)]
        sv_string = sv_string[begin:min(length(sv_string),8)]
        n_step_string = string(n_steps[i])
        println("λ: $λ_string \t energies: $e_string  \t nsteps: $n_step_string \t target: $st_string \t value: $sv_string")
    end
end

run_iron_constrain();