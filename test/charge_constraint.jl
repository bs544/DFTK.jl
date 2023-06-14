include("run_scf_and_compare.jl")
include("testcases.jl")

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [4.0,4.0]
    a = 5.42352
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    model = model_PBE(lattice, atoms, positions;
                      temperature=0.01, magnetic_moments)
    model = convert(Model{Float64}, model)

    idx = 1
    resid_weight = 3.0
    r_sm_frac = 0.05
    α = 0.8

    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[4, 4, 4])
    
    cons_infos = []
    energies = []
    n_steps = []
    V = nothing
    ρ0 = guess_density(basis,magnetic_moments)
    for charge in [1.95,2.0]
        constraints = [DFTK.Constraint(model,idx,resid_weight,r_sm_frac;target_charge=charge)]
        
        scfres = DFTK.scf_potential_mixing(basis; tol=1.0e-8,ρ=ρ0,V=V,constraints=constraints,maxiter=100,damping=DFTK.FixedDamping(α));
        push!(cons_infos,scfres.constraint_info)
        push!(energies,scfres.energies.total)
        push!(n_steps,scfres.n_iter)
        V = DFTK.total_local_potential(scfres.ham)
        ρ0 = scfres.ρ
    end
    println("The zero for λ should correspond to the energy minimum")
    for (i,info) in enumerate(cons_infos)
        λ_string  = string(info[1].λ_charge)
        e_string  = string(energies[i]-energies[1])
        st_string = string(info[1].target_charge)
        sv_string = string(info[1].current_charge)
        
        λ_string  = λ_string[begin:min(length(λ_string),7)]
        e_string  = e_string[begin:min(length(e_string),7)]
        st_string = st_string[begin:min(length(st_string),5)]
        sv_string = sv_string[begin:min(length(sv_string),8)]
        n_step_string = string(n_steps[i])
        println("λ: $λ_string \t energies: $e_string  \t nsteps: $n_step_string \t target: $st_string \t value: $sv_string")
    end
end

run_iron_constrain();