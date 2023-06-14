"""
functions needed to constrain some target properties. e.g. atomic spin

The potential mixing constrained DFT method only requires a change to the residual before preconditioning

The density mixing constrained DFT method is a bit more involved.

NOTE: Charge constraints are harder to converge. Try smaller mixing parameters and smaller weights for the constrained part of the residual
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
    
function integrate_atomic_functions(arr::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints)::Vector{Float64}
    """
    integrate an array arr and the weight functions of each constraint
    """

    rvecs = collect(r_vectors(basis))

    spins = zeros(Float64,length(constraints.cons_vec)) #called spins since this is what you get for integrating the spin density

    for (i,cons) in enumerate(constraints.cons_vec)
        for j in eachindex(rvecs)
            w = weight_fn(rvecs[j],cons)
            if w != 0.0
                spins[i] += w * arr[j]
            end
        end
    end

    spins .*= basis.model.unit_cell_volume / prod(size(rvecs))
    return spins
end

function orthogonalise_residual!(δV::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)
    """
    Orthogonalise the residual with respect to the atomic weight functions
        δV = δV -  ∑ᵢ wᵢ(r) ∑ⱼ (W)ᵢⱼ⁻¹∫δV(r')wⱼ(r')dr'
    """

    rvecs = collect(r_vectors(basis))

    δV_w_i = integrate_atomic_functions(δV,basis,constraints)

    to_be_multiplied_by_weights = constraints.overlap_inv*δV_w_i

    for (i,cons) in enumerate(constraints.cons_vec)
        for j in CartesianIndices(rvecs)
            δV[j] -=  weight_fn(rvecs[j],cons)*to_be_multiplied_by_weights[1][i]
        end
        if is_spin
            cons.λ_spin = to_be_multiplied_by_weights[i]
        else
            cons.λ_charge = to_be_multiplied_by_weights[i]
        end
    end
end      

function add_resid_constraints!(δV::Array{Float64,3},dev_from_target::Vector{Float64},constraints::Constraints,basis::PlaneWaveBasis)
    """
    The part that's added to the residual is ∑ᵢ cᵢ wᵢ(r) ∑ⱼ(W)⁻¹ᵢⱼ (Nⱼ-Nⱼᵗ)
    """
    rvecs = collect(r_vectors(basis))
    W_ij = constraints.overlap_inv
    for (i,cons) in enumerate(constraints.cons_vec)
        factor = cons.cons_resid_weight*(W_ij[i,:]⋅dev_from_target)    
        for j in CartesianIndices(rvecs)
            δV[j] += weight_fn(rvecs[j],cons)*factor
        end
    end
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

function add_constraint_to_residual_component!(δV::Array{Float64,3},ρ::Array{Float64,3},basis::PlaneWaveBasis,constraints::Constraints,is_spin::Bool)

    tmp_constraints = get_spin_charge_constraints(constraints,is_spin)
    if length(tmp_constraints.cons_vec)>0
        current_vals = integrate_atomic_functions(ρ,basis,tmp_constraints)

        if is_spin
            current_vals .*= 2.0 #multiply by 2 to get it in Bohr Magnetons
        end

        for (i,cons) in enumerate(tmp_constraints.cons_vec)
            if is_spin
                cons.current_spin = current_vals[i]
                current_vals[i] -= cons.target_spin
            else
                cons.current_charge = current_vals[i]
                current_vals[i] -= cons.target_charge
            end
        end

        orthogonalise_residual!(δV,basis,tmp_constraints,is_spin)
        add_resid_constraints!(δV,current_vals,tmp_constraints,basis)
    end
end

function add_constraint_to_residual!(δV::Array{Float64,4},ρ::Array{Float64,4}, basis::PlaneWaveBasis,constraints::Constraints)

    δV_charge = δV[:,:,:,1]+δV[:,:,:,2]
    δV_spin   = δV[:,:,:,1]-δV[:,:,:,2]

    ρ_charge = total_density(ρ)
    ρ_spin = spin_density(ρ)

    add_constraint_to_residual_component!(δV_charge,ρ_charge,basis,constraints,false)
    add_constraint_to_residual_component!(δV_spin,  ρ_spin  ,basis,constraints,true )

    δV[:,:,:,1] = 0.5.*(δV_charge + δV_spin)
    δV[:,:,:,2] = 0.5.*(δV_charge - δV_spin)
end


