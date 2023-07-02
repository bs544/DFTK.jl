"""
This is to provide an optimised set of Lagrange multipliers given a trial density. 
This is done by defining functions that give E and ∂E/∂λᵢ and plugging that into Optim.
The point of this is to provide a comparison between inner loop methods and the density mixing cDFT implementation
"""

energy_from_lagrange(lambdas;basis,ρ,weights,ψ,occupation,εF,eigenvalues) = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).energies.total

function lagrange_gradient(lambdas;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol)
    ham = energy_hamiltonian(basis, ψ, occupation; ρ, eigenvalues, εF, cons_lambdas=lambdas, cons_weights=weights).ham
    ρout = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol).ρout
    deriv_array = residual(ρout,ArrayAndConstraints(ρ,lambdas,weights),basis).lambdas
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

    fn(λ) = energy_from_lagrange(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues)
    grad_fn(λ) = lagrange_gradient(λ;basis,ρ,weights,ψ,occupation,εF,eigenvalues,nbandsalg,fermialg,tol=diagtol)

    optim_results = Optim.optimize(fn,grad_fn,lambdas,method=ConjugateGradient();
                                   store_trace=true,extended_trace=true,
                                   iterations=max_cons_iter,x_tol=λ_tol)
    new_lambdas= Optim.x_trace(optim_results)[end]

    return new_lambdas
end