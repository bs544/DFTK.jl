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
    # weights        0.35 0.7 1.4 5.0 10.0 15.0  20.0 25.0 30.0 35.0 40.0 45.0 50.0 55.0 60.0 65.0 70.0
    # approximate     131  76  82  48   50   45    39   38   41   38   34   33   36   35   34   33   38
    # noninteracting   98  58  50  41   41   34    33   31   31   32   32   41   39        41        43
    # None             40  36  37 N/A       
    #
    # weights        20.0 22.0 24.0 26.0 28.0 30.0 32.0 34.0
    # noninteracting             31   32   33   31   31   32
    # nonint_w_initial           37   35   32   
    # nonint_w_initial2          33   30   32   34   31   32
    #
    # weights        35.0 37.0 39.0 41.0 43.0 45.0 47.0 49.0 51.0 53.0 55.0
    # approximate      38   37   38   34   36   33   34   33   35   35   34
    # None equivalent                                34   34   35   35   34
    # Hessian for approximate: [56.49325988115757 0.0; 0.0 56.493259881157584]
    # this should mean that the weights for the approximate case are related to the ones for no preconditioning by a factor of ~56.5
    #
    # weights        0.70 0.80 0.90 1.00 1.10
    # None             35   32   37   34   34

    #initial deviation is [0.6267586571150301, -0.02125928079756023] with no initial_lambdas
    #initial_lambdas makes[-0.06445951158315277, 0.27518702318843413] for initial_lambdas of 0.03 -0.01
    #trying initial_lambdas as 0.03 and 0.0 get [-0.05219991862356199, 0.034682153521223835]
    for resid_weight in [45.0]
        detail = "None"
        resid_weight /= 56.49325988115757
        spin = 1.5
        constraints = [DFTK.Constraint(model,idx,resid_weight,r_sm_frac;target_spin=spin,r_cut),DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin+0.1,r_cut)]
        constraint_term = DFTK.DensityMixingConstraint(constraints)
        terms = model.term_types
        add_constraint!(terms,constraint_term)
        model = Model(model;terms)
        initial_lambdas = nothing #[0.0 0.03; 0.0 0.0]
        basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
        solver = cDFT_anderson_solver(basis)
        # solver = scf_anderson_solver()
        # resid_weight *= basis.dvol
        # constraints = DFTK.get_constraints(basis)
        # c1_arr = constraints.at_fn_arrays[1,2]
        # c2_arr = constraints.at_fn_arrays[2,2]
        # println(c1_arr[1,1,:])
        # println(c2_arr[1,1,:])
        ρ0 = guess_density(basis,magnetic_moments)
        scfres = DFTK.density_mixed_constrained(basis; solver=solver, tol=1.0e-10,ρ=ρ0,maxiter=100,damping=α,lambdas_preconditioning=detail,initial_lambdas)
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