include("run_scf_and_compare.jl")
include("testcases.jl")
using JLD2
using GLMakie
GLMakie.activate!(inline=false)

function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ
    function central_diff(x,y)
        δx = x[2]-x[1]
        y_1 = y[begin:end-2]
        y_2 = y[begin+2:end]
    
        δy = y_2-y_1
    
        grads = δy./(2*δx)
        return x[begin+1:end-1],grads
    end

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

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [4.0,3.0]
    a = 5.42352
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    temperature=0.05
    
    idx = 1
    resid_weight = 0.05 # 0.5 works for simple mixing with α=0.2
    r_sm_frac    = 0.1
    r_cut        = 2.0
    α            = 0.8
    charge       = 4.736
    spin = 1.5    
    detail = "noninteracting"

    model = model_PBE(lattice, atoms, positions;
                      temperature, magnetic_moments)
    
    model = convert(Model{Float64}, model)

    constraints = [DFTK.Constraint(model,idx,resid_weight,r_sm_frac;target_spin=spin,r_cut),DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin+0.1,r_cut)]
    constraint_term = DFTK.DensityMixingConstraint(constraints)
    terms = model.term_types
    add_constraint!(terms,constraint_term)
    model = Model(model;terms)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    constraints=DFTK.get_constraints(basis)
    initial_lambdas = nothing #[0.0 0.008; 0.0 0.0078]
    fermialg=DFTK.default_fermialg(basis.model)
    nbandsalg=AdaptiveBands(basis.model)
    eigensolver=lobpcg_hyper
    tol=0.003
    # λ1_range =  0.025:0.00125:0.035
    # λ2_range = -0.005:0.00125:0.005
    λ1_range = -0.2:0.02:0.1
    λ2_range = -0.2:0.02:0.1
    λ1_Hessian = 0.0
    λ2_Hessian = -0.08
    maxiter = 1
    mixing=DFTK.LdosMixing()

    scfres_fname="test/lambdagrid_initial.jld2"

    #get initial density
    if isfile(scfres_fname)
        scfres = load_scfres(scfres_fname)
    else
        ρ0 = guess_density(basis,magnetic_moments)
        scfres = DFTK.density_mixed_constrained(basis;tol=1.0e-10,ρ=ρ0,maxiter,damping=α,lambdas_preconditioning=detail,initial_lambdas)
        save_scfres(scfres_fname,scfres)
        println("done initial setup")
    end
    ψ_initial = scfres.ψ
    ρ_initial = scfres.ρ
    eigenvalues_initial = scfres.eigenvalues
    occupation_initial = scfres.occupation
    εF_initial = scfres.εF

    #get the wavefunctions etc. for a set of Lagrange multipliers and store them in a named tuple for saving/loading
    results = []
    λ2 = 0.0
    for λ1 in λ1_range
        
        for λ2 in λ2_range
            println(λ1,"\t",λ2)
            lambdas = [0.0 λ1; 0.0 λ2]
            ham = energy_hamiltonian(basis, ψ_initial, occupation_initial; ρ=ρ_initial, eigenvalues=eigenvalues_initial, εF=εF_initial, cons_lambdas=lambdas).ham
            ψ, eigenvalues, occupation, εF, ρout = DFTK.next_density(ham, nbandsalg, fermialg; eigensolver, ψ=ψ_initial, eigenvalues=eigenvalues_initial,
                                                        occupation=occupation_initial, miniter=1, tol)
            E = DFTK.W_λ(ham,basis,ρout,ρ_initial,ψ,occupation,eigenvalues,εF,lambdas)
            λ_grad = DFTK.get_density_deviation(ρout,basis)
            λ_grad = [λ_grad[1,2],λ_grad[2,2]]
            if λ1 == λ1_Hessian || λ2 == λ2_Hessian
                Hessian = DFTK.constrained_second_deriv(ham,basis,εF,eigenvalues,occupation,ψ,constraints,ρout,maxiter,mixing,detail)
            else
                Hessian = nothing #
            end
            #the hessian calculation may need the following:
            # ham,basis,εF,eigenvalues,occupation,ψ,constraints,ρin,n_iter,mixing,detail
            # ham,εF,eigenvalues,occupation and ψ are specific to this set of Lagrange multipliers, the rest are more general
            scfres_specific=(;ρout,E,λ_grad,lambdas,Hessian)
            push!(results,scfres_specific)
        end
    end
        

    res = results
    λ1s = collect(λ1_range)
    λ2s = collect(λ2_range)
    E_arr = zeros(Float64,length(λ1s),length(λ2s))
    λs = zeros(Float64,length(λ1s),length(λ2s),2)
    E_derivs = zeros(Float64,length(λ1s),length(λ2s),2)
    Hessians = zeros(Float64,length(λ1s),length(λ2s),2,2)
    λ1_idx = findfirst(x->x==λ1_Hessian,λ1s)
    λ2_idx = findfirst(x->x==λ2_Hessian,λ2s)
    for (i,λ1) in enumerate(λ1_range)
        for (j,λ2) in enumerate(λ2_range)
            idx = j + (i-1)*length(λ1s)
            λ_ele = res[idx].lambdas
            λs[i,j,:] = [λ_ele[1,2],λ_ele[2,2]]
            E_arr[i,j] = res[idx].E
            E_derivs[i,j,:] = res[idx].λ_grad
            if i == λ1_idx || j == λ2_idx
                Hessians[i,j,:,:] = res[idx].Hessian
            end
        end
    end

    #get variation of dE/dλ₁ wrt λ₁ and λ₂, and do the same for dE/dλ₂
    
    x_11,H_11 = central_diff(λ1s,E_derivs[:,λ2_idx,1])
    x_12,H_12 = central_diff(λ1s,E_derivs[:,λ2_idx,2])
    x_21,H_21 = central_diff(λ2s,E_derivs[λ1_idx,:,1])
    x_22,H_22 = central_diff(λ2s,E_derivs[λ1_idx,:,2])

    H_11_analytic = Hessians[:,λ2_idx,1,1].*-1
    H_12_analytic = Hessians[:,λ2_idx,1,2].*-1
    H_21_analytic = Hessians[λ1_idx,:,2,1].*-1
    H_22_analytic = Hessians[λ1_idx,:,2,2].*-1


    f = Figure()
    # ax = Axis(f[1,1])
    # lines!(ax,vec(λs[:,:,1]),vec(E_arr))
    ax = Axis3(f[1,1])
    surface!(ax,vec(λs[:,:,1]),vec(λs[:,:,2]),vec(E_arr))
    ax_arr = Axis(f[1,2])
    arrows!(ax_arr,vec(λs[:,:,1]),vec(λs[:,:,2]),vec(E_derivs[:,:,1]),vec(E_derivs[:,:,2]),lengthscale=0.025)
    ax_Hess_diag = Axis(f[2,1])
    ax_Hess_off  = Axis(f[2,2])

    lines!(ax_Hess_diag,x_11,H_11,label="H_11 numeric")
    lines!(ax_Hess_diag,λ1s,H_11_analytic,label="H_11 analytic")
    lines!(ax_Hess_diag,x_22,H_22,label="H_22 numeric")
    lines!(ax_Hess_diag,λ2s,H_22_analytic,label="H_22 analytic")
    axislegend(ax_Hess_diag)

    lines!(ax_Hess_off,x_12,H_12,label="H_12 numeric")
    lines!(ax_Hess_off,λ1s,H_12_analytic,label="H_12 analytic")
    lines!(ax_Hess_off,x_21,H_21,label="H_21 numeric")
    lines!(ax_Hess_off,λ2s,H_21_analytic,label="H_21 analytic")
    axislegend(ax_Hess_off)

    display(f)



        


    

end

run_iron_constrain();