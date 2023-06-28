include("scf_callbacks.jl")

# Struct to store some options for forward-diff / reverse-diff response
# (unused in primal calculations)
@kwdef struct ResponseOptions
    verbose = false
end

"""
Obtain new density ρ by diagonalizing `ham`. Follows the policy imposed by the `bands`
data structure to determine and adjust the number of bands to be computed.
"""
function next_density(ham::Hamiltonian,
                      nbandsalg::NbandsAlgorithm=AdaptiveBands(ham.basis.model),
                      fermialg::AbstractFermiAlgorithm=default_fermialg(ham.basis.model);
                      eigensolver=lobpcg_hyper, ψ=nothing, eigenvalues=nothing,
                      occupation=nothing, kwargs...)
    n_bands_converge, n_bands_compute = determine_n_bands(nbandsalg, occupation,
                                                          eigenvalues, ψ)

    if isnothing(ψ)
        increased_n_bands = true
    else
        @assert length(ψ) == length(ham.basis.kpoints)
        n_bands_compute = max(n_bands_compute, maximum(ψk -> size(ψk, 2), ψ))
        increased_n_bands = n_bands_compute > size(ψ[1], 2)
    end

    # TODO Synchronize since right now it is assumed that the same number of bands are
    #      computed for each k-Point
    n_bands_compute = mpi_max(n_bands_compute, ham.basis.comm_kpts)

    eigres = diagonalize_all_kblocks(eigensolver, ham, n_bands_compute;
                                     ψguess=ψ, n_conv_check=n_bands_converge, kwargs...)
    eigres.converged || (@warn "Eigensolver not converged" iterations=eigres.iterations)

    # Check maximal occupation of the unconverged bands is sensible.
    occupation, εF = compute_occupation(ham.basis, eigres.λ, fermialg;
                                        tol_n_elec=nbandsalg.occupation_threshold)
    minocc = maximum(minimum, occupation)

    # TODO This is a bit hackish, but needed right now as we increase the number of bands
    #      to be computed only between SCF steps. Should be revisited once we have a better
    #      way to deal with such things in LOBPCG.
    if !increased_n_bands && minocc > nbandsalg.occupation_threshold
        @warn("Detected large minimal occupation $minocc. SCF could be unstable. " *
              "Try switching to adaptive band selection (`nbandsalg=AdaptiveBands(model)`) " *
              "or request more converged bands than $n_bands_converge (e.g. " *
              "`nbandsalg=AdaptiveBands(model; n_bands_converge=$(n_bands_converge + 3)`)")
    end

    ρout = compute_density(ham.basis, eigres.X, occupation; nbandsalg.occupation_threshold)
    (; ψ=eigres.X, eigenvalues=eigres.λ, occupation, εF, ρout, diagonalization=eigres,
     n_bands_converge, nbandsalg.occupation_threshold)
end


@doc raw"""
    self_consistent_field(basis; [tol, mixing, damping, ρ, ψ])

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
"""
@timing function self_consistent_field(
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
    function fixpoint_map(ρin)
        converged && return ρin  # No more iterations if convergence flagged
        n_iter += 1

        # Note that ρin is not the density of ψ, and the eigenvalues
        # are not the self-consistent ones, which makes this energy non-variational
        if typeof(ρin)==ArrayAndConstraints
          energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin.arr, eigenvalues, εF, cons_λ=ρin.λ,cons_weight=ρin.weight)
        else
          energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρin, eigenvalues, εF)
        end


        # Diagonalize `ham` to get the new state
        nextstate = next_density(ham, nbandsalg, fermialg; eigensolver, ψ, eigenvalues,
                                 occupation, miniter=1, tol=determine_diagtol(info))
        ψ, eigenvalues, occupation, εF, ρout = nextstate

        if typeof(ρin)==ArrayAndConstraints
          # case when constrained dft is happening
          # adding this to make sure that when the residual (ρout-ρin) is calculated, it includes the gradient of the energy wrt the lagrange multipliers
          ρout = residual(ρout,ρin,basis) + ρin
        end


        # Update info with results gathered so far
        info = (; ham, basis, converged, stage=:iterate, algorithm="SCF",
                ρin, ρout, α=damping, n_iter, nbandsalg.occupation_threshold,
                nextstate..., diagonalization=[nextstate.diagonalization])
        #for some reason the ρout field isn't in the ArrayAndConstraints type,
        #so I'm readding it here to see if that helps
        info = merge(info,(;ρout))

        # Compute the energy of the new state
        if compute_consistent_energies
            if typeof(ρout)==ArrayAndConstraints
              energies = energy_hamiltonian(basis, ψ, occupation;
                                            ρ=ρout.arr, eigenvalues, εF, cons_λ=ρout.λ,cons_weight=ρout.weight).energies
            else
              energies = energy_hamiltonian(basis, ψ, occupation;
                                          ρ=ρout, eigenvalues, εF).energies
            end
        end
        info = merge(info, (; energies))
        # Apply mixing and pass it the full info as kwargs
        #This mixing is the preconditioning of the density residual
        #I don't think the constraints need to be preconditioned in the same way, so just pass the array
        if typeof(ρout)==ArrayAndConstraints
          R = ρout.arr-ρin.arr
          info = merge(info,(;ρin=ρin.arr,ρout=ρout.arr))#in case of χ0 mixing
          δρ = mix_density(mixing,basis,R; info...)
          info = merge(info,(;ρin,ρout))#make the density ArrayAndConstraints again
          δρ = ArrayAndConstraints(δρ,ρout.λ-ρin.λ,ρout.weight)

          ρnext = ρin + T(damping)*δρ
        else
          δρ = mix_density(mixing, basis, ρout - ρin ; info...)
          ρnext = ρin .+ T(damping) .* δρ
        end
        info = merge(info, (; ρnext))
        

        callback(info)
        converged = is_converged(info)
        converged = MPI.bcast(converged, 0, MPI.COMM_WORLD)  # Ensure same converged

        ρnext
    end

    # Tolerance and maxiter are only dummy here: Convergence is flagged by is_converged
    # inside the fixpoint_map.
    solver(fixpoint_map, ρout, maxiter; tol=eps(T))

    # We do not use the return value of solver but rather the one that got updated by fixpoint_map
    # ψ is consistent with ρout, so we return that. We also perform a last energy computation
    # to return a correct variational energy
    if typeof(ρout)==ArrayAndConstraints
      energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout.arr, eigenvalues, εF,cons_λ=ρout.λ,cons_weight=ρout.weight)
    else
      energies, ham = energy_hamiltonian(basis, ψ, occupation; ρ=ρout, eigenvalues, εF)
    end

    # Measure for the accuracy of the SCF
    # TODO probably should be tracked all the way ...
    norm_Δρ = norm(info.ρout - info.ρin) * sqrt(basis.dvol)

    # Callback is run one last time with final state to allow callback to clean up
    info = (; ham, basis, energies, converged, nbandsalg.occupation_threshold,
            ρ=ρout, α=damping, eigenvalues, occupation, εF, info.n_bands_converge,
            n_iter, ψ, info.diagonalization, stage=:finalize,
            algorithm="SCF", norm_Δρ)
    callback(info)
    info
end
