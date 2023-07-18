include("run_scf_and_compare.jl")
include("testcases.jl")
using Plots
using HDF5

function finite_diff(x,y)
    x_1 = x[begin:end-1]
    x_2 = x[begin+1:end]
    y_1 = y[begin:end-1]
    y_2 = y[begin+1:end]

    dys = y_2-y_1
    dxs   = x_2-x_1
    mid_xs= x_1+0.5.*dxs

    grads = dys./dxs
    return mid_xs,grads
end

function central_diff(x,y)
    δx = x[2]-x[1]
    y_1 = y[begin:end-2]
    y_2 = y[begin+2:end]

    δy = y_2-y_1

    grads = δy./(2*δx)
    return x[begin+1:end-1],grads
end

function get_scf(model;ecut,kgrid,magnetic_moments,tol=1.0e-10,maxiter=100,α=0.8,fname)
    basis = PlaneWaveBasis(model;Ecut=ecut,kgrid=kgrid)
    ρ0 = guess_density(basis,magnetic_moments)
    scfres = DFTK.self_consistent_field(basis;ρ=ρ0,maxiter,tol,damping=α)
    h5open(fname,"w") do file
        write(file,"rho",ρ0)
    end
    return scfres.ρ
end


function run_iron_constrain()
    #TODO: Double check units for this. I think they're off somewhere. Maybe the spin density isn't in μᵦ

    # Produce reference data and guess for this configuration
    Fe = ElementPsp(iron_bcc.atnum, psp=load_psp("hgh/lda/Fe-q8.hgh"))
    atoms, positions = [Fe,Fe], [zeros(3),0.5.*ones(3)]
    magnetic_moments = [2.0,2.0]
    a = 5.42352
    lattice = a .* [1.0 0.0 0.0;
                    0.0 1.0 0.0;
                    0.0 0.0 1.0]
    model = model_PBE(lattice, atoms, positions;
                      temperature=0.01, magnetic_moments,symmetries=false)
    model = convert(Model{Float64}, model)


    resid_weight = 1.0#0.01
    r_sm_frac = 0.1
    r_cut = 2.0
    spin = 1.5
    charge = 4.736
    spin_cons = true
    if spin_cons
        idx = 2
        target = spin
        fname = "./test/constraint_spin_data_iron_pbe_rho.h5"
        constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=target,target_charge=charge,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    else
        idx = 1
        target = charge
        fname = "./test/constraint_charge_data_iron_pbe_rho.h5"
        constraints     = [DFTK.Constraint(model,1,resid_weight,r_sm_frac;target_spin=spin,target_charge=target,r_cut)]#,DFTK.Constraint(model,2,resid_weight,r_sm_frac;target_spin=spin,r_cut)]
    end

    if !isfile(fname)
        ρ0 = get_scf(model;ecut=15,kgrid=[3,3,3],magnetic_moments,fname)
    else
        ρ0 = h5open(fname,"r") do file
            read(file,"rho")
        end
    end

    # basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    lambdas   = [i for i ∈-0.1:0.01:0.0]#0.1]
    Ws        = []
    grads     = []
    Ns        = []
    gradgrads = []

    constraint_term = DFTK.DensityMixingConstraint(constraints)
    
    terms = model.term_types
    push!(terms,constraint_term)
    model = Model(model;terms)
    basis = PlaneWaveBasis(model; Ecut=15, kgrid=[3,3,3])
    
    # ρ0     = guess_density(basis,magnetic_moments)
    ρ_cons = DFTK.ArrayAndConstraints(ρ0,basis)
    
    diagtol   = 0.003
    nbandsalg = AdaptiveBands(basis.model)
    fermialg  = DFTK.default_fermialg(basis.model)
    
    λ0 = 0.1
    for lambda in lambdas
        println("lambda: $lambda")
        energy,grad,hessian = DFTK.EnDerivsFromLagrange([lambda,λ0];basis,ρ=ρ0,weights=ρ_cons.weights,tol=diagtol,
                                              ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,nbandsalg,fermialg,constraints=DFTK.get_constraints(basis))
        N = grad[1]
        N = N + target

        println(hessian)

        push!(Ws,energy)
        push!(grads,grad[2])
        push!(gradgrads,hessian[1,2])
        push!(Ns,N)
    end

    fd_x,fd_grads = central_diff(lambdas,Ws)
    fd_x,fd_hess = central_diff(lambdas,grads)
    # grad_diffs = (grads[begin+1:end-1]-fd_E)./fd_x

    p1 = plot(lambdas,Ws.-Ws[1],label="W")
    plot!(p1,legend=:bottomleft)
    
    display(p1)

    p2 = plot(lambdas,grads,label="N-Nᵗ")
    plot!(p2,fd_x,fd_grads ,label="dW/dλ")
    display(p2)

    p3 = plot(lambdas,gradgrads,label="H_analytic")
    plot!(p3,fd_x,fd_hess,label="H_fd")
    display(p3)



end

run_iron_constrain();