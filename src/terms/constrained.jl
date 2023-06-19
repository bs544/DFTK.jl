"""
Adds the constrained potential to the hamiltonian given the modified density struct ArrayAndConstraints
Also overloads the other ene_ops functions so that they can take in this struct so that it may be passed into energy_hamiltonian
"""

mutable struct Constraint
    """
    This is supposed to define the constraint on the spin and/or charge around an atom
    Just use one of these per atom
    """
    atom_pos          :: Vector{Float64}
    atom_idx          :: Int     #index of atom as specified in model
    spin              :: Bool
    charge            :: Bool
    r_sm              :: Float64 #cutoff radius for atomic function
    r_cut             :: Float64 #smearing width for atomic function
    target_spin       :: Float64
    target_charge     :: Float64
    current_spin      :: Float64
    current_charge    :: Float64
    cons_resid_weight :: Float64
    λ_charge          :: Float64 #Lagrange multiplier for constraint
    λ_spin            :: Float64
end

function Constraint(model::Model,idx::Int,cons_resid_weight::Float64=1.0,r_sm_frac::Float64=0.05;target_spin=nothing,target_charge=nothing)::Constraint
    atom_pos = model.positions[idx]
    psp = model.atoms[idx].psp
    r_cut = maximum(psp.rp)
    r_cut = max(r_cut,psp.rloc)
    r_sm = r_cut*r_sm_frac
    spin = true
    charge = true
    if isnothing(target_spin)
        target_spin = 0.0
        spin = false
    end
    if isnothing(target_charge)
        target_charge = 0.0
        charge = false
    end
    @assert charge || spin
    return Constraint(atom_pos,idx,spin,charge,r_sm,r_cut,target_spin,target_charge, 0.0, 0.0, cons_resid_weight,0.0,0.0)
end

function calculate_overlap(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Array{Float64,2}
    """
    integrate the overlap between the individual atomic weight functions Wᵢⱼ = ∫wᵢ(r)wⱼ(r)d
    """
    W_ij = zeros(Float64,length(cons_vec),length(cons_vec))
    overlap = zeros(Bool,length(cons_vec),length(cons_vec))
    r_vecs = collect(r_vectors(basis))

    #check if there's any overlap
    for i = 1:length(cons_vec)-1
        for j = i+1:length(cons_vec)
            r_ij = cons_vec[i].atom_pos - cons_vec[j].atom_pos
            dist = sqrt(sum(r_ij.^2))
            if dist ≤ cons_vec[i].r_cut+cons_vec[j].r_cut
                overlap[i,j] = 1
                overlap[j,i] = 1
            end
        end
    end

    #do the diagonal terms either way
    for (i,cons) in enumerate(cons_vec)
        for r in r_vecs
            W_ij[i,i] += weight_fn(r,cons)^2
        end
    end

    #if there's no overlap, then that's it. Otherwise add the off diagonal terms as necessary
    if sum(overlap) != 0
        for i = 1:length(cons_vec)
            for j = 1:length(cons_vec)
                if overlap[i,j]
                    for r in r_vecs
                        W_ij[i,j] += weight_fn(r,cons_vec[i])*weight_fn(r,cons_vec[j])
                    end
                end
            end
        end
    end
    return W_ij .* (basis.model.unit_cell_volume/prod(size(r_vecs)))
end

struct Constraints
    cons_vec    :: Vector{Constraint}
    overlap     :: Array{Float64,2} #overlap matrix for the different atomic functions
    overlap_inv :: Array{Float64,2}
end

function Constraints(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Constraints
    overlap = calculate_overlap(cons_vec,basis)
    overlap_inv = inv(overlap)
    return Constraints(cons_vec,overlap,overlap_inv)
end

function get_spin_charge_constraints(constraints::Constraints,is_spin::Bool)::Constraints
    """
    Remove the constraints that don't constrain the specified variable (either charge or spin) defined by is_spin
    Modify the overlap matrix and the inverse overlap matrix to account as needed for the given 
    """
    
    relevant_constraints = Vector{Int}(undef,0)
    relevant_cons_vec = Vector{Constraint}(undef,0)
    for (i,cons) in enumerate(constraints.cons_vec)
        if (is_spin && cons.spin) || (!is_spin && cons.charge)
            push!(relevant_constraints,i)
            push!(relevant_cons_vec,cons)
        end
    end
    if length(relevant_cons_vec)>0
        relevant_overlap = constraints.overlap[relevant_constraints,relevant_constraints]
        relevant_overlap_inv = inv(relevant_overlap)
    else
        relevant_overlap = Array{Float64,2}(undef,0,0)
        relevant_overlap_inv = Array{Float64,2}(undef,0,0)
    end
    new_constraints = Constraints(relevant_cons_vec,relevant_overlap,relevant_overlap_inv)
    return new_constraints
end

mutable struct ArrayAndConstraints
    """
    Struct to combine either the residual or the density with the constraints
    """
    arr         :: Array{Float64,4}
    λ           :: Array{Float64,2} #first dimension for constraints, second defines either spin or charge
    weight      :: Array{Float64,2} #define based on both cons_resid_weight and the unit cell volume
end

function ArrayAndConstraints(arr::Array{Float64,4},constraints::Constraints)::ArrayAndConstraints
    λ = zeros(Float64,length(constraints.cons_vec),2)
    weight = ones(Float64,size(λ))
    for (i,cons) in enumerate(constraints.cons_vec)
        weight[i,:] .*= cons.unit_cell_volume*cons.cons_resid_weight
    end
    return ArrayAndConstraints(arr,λ,weight)
end

Base.:+(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr + b.arr,a.λ+b.λ,a.weight)
Base.:-(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr - b.arr,a.λ-b.λ,a.weight)
Base.vec(a::ArrayAndConstraints) = vcat(vec(a.arr),vec(a.λ .* a.weight))
LinearAlgebra.norm(a::ArrayAndConstraints) = norm(a.arr) + norm(a.λ .* a.weight)
Base.:*(a::Float64,b::ArrayAndConstraints) = ArrayAndConstraints(a.*b.arr,a.*b.λ,b.weight)

ene_ops(args...;ρ::ArrayAndConstraints,kwargs...) = ene_ops(args...;ρ.arr,kwargs...)

function weight_fn(r::AbstractVector{Float64},cons::Constraint)::Float64
    at_r = sqrt(sum((r - cons.atom_pos).^2))
    if at_r > cons.r_cut
        return 0.0
    elseif at_r < cons.r_cut-cons.r_sm
        return 1.0
    else
        x = (cons.r_cut-at_r)/cons.r_sm
        return x^2 * (3 + x*(1 + x*( -6 + 3*x)))
    end
end


struct DensityConstraint end

function add_constraint_to_arr!(arr::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)
    rvecs = collect(r_vectors(basis))
    if is_spin
        λs = [cons.λ_spin   for cons in constraints.cons_vec]
    else
        λs = [cons.λ_charge for cons in constraints.cons_vec]
    end

    for (i,cons) in enumerate(constraints.cons_vec)
        factor = constraints.overlap_inv[i,:]⋅λs
        for j in CartesianIndices(rvecs)
            arr[j] += weight_fn(rvecs[j],cons)*factor
        end
    end
end

struct TermDensityConstraint 
    constraints::Constraints
end


@timing "ene_ops: constraint" function ene_ops(term::TermDensityConstraint,basis::PlaneWaveBasis;ρ::ArrayAndConstraints)
    #update the constraints
    for (i,cons) in enumerate(term.constraints.cons_vec)
        cons.λ_charge = ρ.λ[i,1]
        cons.λ_spin = ρ.λ[i,2]
    end
    #define the constrained potential array
    pot_size = basis.fft_size
    if basis.model.spin_polarization==:collinear
        pot_size = (pot_size...,2)
    end
    constrained_potential=zeros(Float64,pot_size)
    

    charge_constraint_pot = zeros(Float64,basis.fft_size)
    charge_constraints = get_spin_charge_constraints(term.constraints,false)
    add_constraint_to_arr!(charge_constraint_pot,basis,charge_constraints,false)


    if length(pot_size)==4 # then spin and charge can be constrained, so do the spin too
        
        spin_constraint_pot = zeros(Float64,basis.fft_size)
        spin_constraints   = get_spin_charge_constraints(term.constraints,true)
        add_constraint_to_arr!(spin_constraint_pot,basis,spin_constraints,true)

        constrained_potential[:,:,:,1] = 0.5.*(charge_constraint_pot+spin_constraint_pot)
        constrained_potential[:,:,:,2] = 0.5.*(charge_constraint_pot-spin_constraint_pot)
    else
        constrained_potential = charge_constraint_pot
    end

    ops = [RealSpaceMultiplication(basis, kpt, constrained_potential[:,:,:,kpt.spin])
           for kpt in basis.kpoints]
    if :ρ in keys(kwargs)
        E = sum(total_density(ρ.arr) .* constrained_potential) * basis.dvol
    else
        E = T(Inf)
    end

    (; E, ops)

end




