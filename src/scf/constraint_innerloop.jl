"""
This is to provide an optimised set of Lagrange multipliers given a trial density. 
This is done by defining functions that give E and ∂E/∂λᵢ and plugging that into Optim.
The point of this is to provide a comparison between inner loop methods and the density mixing cDFT implementation
"""

function lambdas_2_vector(lambdas,constraints::Constraints)
    """
    For the purposes of minimisation, the lambdas array needs to be a vector of only the elements that are being constrained
    """
    T = typeof(lambdas[1])
    λ = Vector{T}(undef,sum(constraints.is_constrained))
    idx = 1
    for i in eachindex(lambdas)
        if constraints.is_constrained[i]==1
            λ[idx] = lambdas[i]
            idx +=1
        end
    end
    @assert idx-1 == sum(constraints.is_constrained)
    return λ
end

function vector_2_lambdas(λ,constraints::Constraints)
    """
    turn the vector back to the array
    """
    lambdas = zeros(Number,size(constraints.is_constrained))
    idx = 1
    for i in eachindex(lambdas)
        if constraints.is_constrained[i]==1
            lambdas[i]= λ[idx]
            idx += 1
        end
    end
    return lambdas
end

function EnGradFromLagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,eigensolver=lobpcg_hyper,nbandsalg,fermialg,tol,constraints)

    lambdas = vector_2_lambdas(λ,constraints)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ψ, eigenvalues, occupation, εF, ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                        occupation, miniter=1, tol)
    E = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF,cons_lambdas=lambdas,cons_weights=weights).energies

    deriv_array = residual(ρout,ArrayAndConstraints(ρ,lambdas,weights),basis).lambdas
    deriv_array ./= weights
    deriv_array = lambdas_2_vector(deriv_array,constraints)
            
    return E, deriv_array
end

function dielectric_operator(δρ,basis,ρ,ham,ψ,occupation,εF,eigenvalues)
    δV = apply_kernel(basis,δρ;ρ)
    χ0δV = apply_χ0(ham,ψ,occupation,εF,eigenvalues,δV)
    return δρ - χ0δV
end

function second_deriv_wrt_lagrange(λ,constraints,interacting=true;basis,ρ,ham,ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing,tol=nothing,nbandsalg=nothing,fermialg=nothing,eigensolver=lobpcg_hyper)

    lambdas = vector_2_lambdas(λ,constraints)
    weights = constraints.res_wgt_arrs
    if nothing in [ψ,occupation,εF,eigenvalues]
        #assume ρ is ρin
        #need to generate a hamiltonian to diagonalise
        if isnothing(ham)
            ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
        end
        ψ, eigenvalues, occupation, εF, ρ = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                         occupation, miniter=1, tol)
    end

    ε(arr) = dielectric_operator(arr,basis,ρ,ham,ψ,occupation,εF,eigenvalues)

    at_fn_arrs = get_4d_at_fns(constraints)
    at_fns = lambdas_2_vector(at_fn_arrs,constraints)
    χ0_at_fns = [apply_χ0(ham,ψ,occupation,εF,eigenvalues,at_fn) for at_fn in at_fns]
    if interacting
        inv_εs = [linsolve(arr->ε(arr),χ0_at_fn,verbosity=1)[1] for χ0_at_fn in χ0_at_fns]
    else
        inv_εs = χ0_at_fns
    end

    Hessian = zeros(Float64,length(at_fns),length(at_fns))
    
    for i = 1:length(at_fns)
        for j = i:length(at_fns)
            Hessian[i,j] = sum(at_fns[i].*inv_εs[j])*constraints.dvol
            Hessian[j,i] = Hessian[i,j]
        end
    end

    return Hessian
end

function EnDerivsFromLagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,eigensolver=lobpcg_hyper,nbandsalg,fermialg,tol,constraints)

    lambdas = vector_2_lambdas(λ,constraints)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ψ, eigenvalues, occupation, εF, ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                        occupation, miniter=1, tol)
    E = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF,cons_lambdas=lambdas,cons_weights=weights).energies

    deriv_array = residual(ρout,ArrayAndConstraints(ρ,lambdas,weights),basis).lambdas
    deriv_array ./= weights
    deriv_array = lambdas_2_vector(deriv_array,constraints)
    
    Hessian = second_deriv_wrt_lagrange(λ,constraints,false;basis,ρ=ρout,ham,ψ,occupation,εF,eigenvalues)

    return E, deriv_array, Hessian
end

function energy_from_lagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,eigensolver=lobpcg_hyper,nbandsalg,fermialg,tol,constraints) 
    """
    convert from the vector to the lambdas array
    then get the hamiltonian with energy_hamiltonian
    then diagonalise the hamiltonian with next_density
    use the outputs of the diagonalisation to get the energy
    """
    lambdas = vector_2_lambdas(λ,constraints)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ψ, eigenvalues, occupation, εF, ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                                        occupation, miniter=1, tol)
    E = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF,cons_lambdas=lambdas,cons_weights=weights).energies
    return E#.total
    # grad = lagrange_gradient(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol,eigensolver,constraints)
    # return dot(grad,grad)
end

function lagrange_gradient(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol,eigensolver=lobpcg_hyper,constraints)
    """
    Convert from the vector to the lambdas array, 
    then get the hamiltonian from the density and lambdas array,
    then get the output density by diagonalising the hamiltonian,
    use this density to get the derivative array (i.e. difference between current values and target values)
    """
    lambdas = vector_2_lambdas(λ,constraints)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol).ρout
    deriv_array = residual(ρout,ArrayAndConstraints(ρ,lambdas,weights),basis).lambdas
    deriv_array ./= weights
    # spins = integrate_atomic_functions(spin_density(ρout),constraints,2)
    # println(spins,constraints.target_values[:,2])
    deriv_array = lambdas_2_vector(deriv_array,constraints)
    return deriv_array
end

function innerloop(ρ_cons::ArrayAndConstraints,basis,nbandsalg,fermialg,diagtol,λ_tol,max_cons_iter;ψ=nothing,occupation=nothing,εF=nothing,eigenvalues=nothing)
    """
    Find the value for the Lagrange Multipliers that satisfies the constraints kept in basis.
    Beyond the expected inputs needed to form and diagonalise the Hamiltonian, the two other inputs are:
        λ_tol         : the tolerance in the values of the λ values below which the optimisation finishes
        max_cons_iter : the maximum number of iterations of this constraining optimisation. I don't know what this should be set to at the moment, surely 10 would do?
    """
    ρ = ρ_cons.arr
    lambdas = ρ_cons.lambdas
    weights = ρ_cons.weights
    constraints = get_constraints(basis)

    ben_fn(λ) = energy_from_lagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol=diagtol,constraints)#.*-1.0
    ben_grad_fn(λ) = lagrange_gradient(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol=diagtol,constraints).*-1.0

    λ = lambdas_2_vector(lambdas,constraints)
    λ = zeros(Float64,size(λ))
    optim_results = Optim.optimize(ben_fn,ben_grad_fn,λ,method=ConjugateGradient();
                                   store_trace=true,extended_trace=true,
                                   iterations=max_cons_iter,show_trace=true,inplace=false,x_tol=λ_tol)
    new_lambdas= Optim.x_trace(optim_results)[end]

    println(optim_results)
    # println(Optim.x_trace(optim_results))
    return vector_2_lambdas(new_lambdas,constraints)
end