@doc raw"""
    density_mixed_constrained(basis; [tol, mixing, damping, ρ, ψ])

Solve the Kohn-Sham equations with a density-based SCF algorithm using damped, preconditioned
iterations where ``ρ_\text{next} = α P^{-1} (ρ_\text{out} - ρ_\text{in})``.

Overview of parameters:
- `ρ`:   Initial density
- `ψ`:   Initial orbitals
- `tol`: Tolerance for the density change (``\|ρ_\text{out} - ρ_\text{in}\|``)
  to flag convergence. Default is `1e-6`.
- `is_converged`: Convergence control callback. Typical objects passed here are
  `DFTK.ScfConvergenceDensity(tol)` (the default), `DFTK.ScfConvergenceEnergy(tol)`
  or `DFTK.ScfConvergenceForce(tol)`.
- `maxiter`: Maximal number of SCF iterations
- `mixing`: Mixing method, which determines the preconditioner ``P^{-1}`` in the above equation.
  Typical mixings are [`LdosMixing`](@ref), [`KerkerMixing`](@ref), [`SimpleMixing`](@ref)
  or [`DielectricMixing`](@ref). Default is `LdosMixing()`
- `damping`: Damping parameter ``α`` in the above equation. Default is `0.8`.
- `nbandsalg`: By default DFTK uses `nbandsalg=AdaptiveBands(model)`, which adaptively determines
  the number of bands to compute. If you want to influence this algorithm or use a predefined
  number of bands in each SCF step, pass a [`FixedBands`](@ref) or [`AdaptiveBands`](@ref).
- `callback`: Function called at each SCF iteration. Usually takes care of printing the
  intermediate state.

My modification:
One of the terms in basis is for constraining the density. These constraints are also important in the mixing of the density,
which is done within a combined struct ArrayAndConstraints
"""
@timing function density_mixed_constrained(
    basis::PlaneWaveBasis{T};
    ρ=guess_density(basis),
    ψ=nothing,
    tol=1e-6,
    is_converged=ScfConvergenceDensity(tol),
    maxiter=100,
    mixing=LdosMixing(),
    damping=0.8,
    solver=scf_anderson_solver(),
    eigensolver=lobpcg_hyper,
    determine_diagtol=ScfDiagtol(),
    nbandsalg::NbandsAlgorithm=AdaptiveBands(basis.model),
    fermialg::AbstractFermiAlgorithm=default_fermialg(basis.model),
    callback=ScfDefaultCallback(; show_damping=false),
    compute_consistent_energies=true,
    response=ResponseOptions(),  # Dummy here, only for AD
) where {T}
    # All these variables will get updated by fixpoint_map
    if !isnothing(ψ)
        @assert length(ψ) == length(basis.kpoints)
    end
    occupation = nothing
    eigenvalues = nothing
    ρout = ρ
    εF = nothing
    n_iter = 0
    energies = nothing
    ham = nothing
    info = (; n_iter=0, ρin=ρ)  # Populate info with initial values
    converged = false

    # We do density mixing in the real representation
    # TODO support other mixing types
    function fixpoint_map(ρin_cons)
        #ρin_cons: ArrayAndConstraints, contains the density and the constraints
        #ρin     : Density Array
        converged && return ρin_cons  # No more iterations if convergence flagged
        n_iter += 1

        ρin = ρin_cons.arr

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF, 
                                          cons_lambdas=ρin_cons.lambdas, cons_weights=ρin_cons.weights)

        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol=determine_diagtol(info))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        resid = residual(ρout,ρin_cons,basis)
        println(resid.lambdas)
        ρout_cons = resid + ρin_cons

        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, nbandsalg.occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization],
                ρin_cons, ρout_cons)

        # Compute the energy of the new state
        if compute_consistent_energies
            energies = energy_hamiltonian(basis, ψ, occupation;
                                          ρ=ρout, eigenvalues, εF, 
                                          cons_lambdas=ρout_cons.lambdas, cons_weights=ρout_cons.weights).energies
        end
        info = merge(info, (; energies))

        # Apply mixing and pass it the full info as kwargs
        δρ = mix_density(mixing, basis, ρout - ρin; info...)
        δρ_cons = ArrayAndConstraints(δρ,resid.lambdas,resid.weights)
        ρnext_cons = ρin_cons .+ T(damping) .* δρ_cons
        info = merge(info, (; ρnext_cons))

        callback(info)
        converged = is_converged(info)
        converged = MPI.bcast(converged, 0, MPI.COMM_WORLD)  # Ensure same converged

        ρnext_cons
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map.
    ρout_cons = ArrayAndConstraints(ρout,basis)
    solver(fixpoint_map, ρout_cons, maxiter; tol=eps(T))

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout_cons.arr, eigenvalues, εF, 
                                       ρout_cons.lambdas, ρout_cons.weights)

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout_cons - info.ρin_cons) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
            ρ=ρout, α=damping, eigenvalues, occupation, εF, info.n_bands_converge,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ)
    callback(info)
    info
end
