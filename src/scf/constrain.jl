"""
functions needed to constrain some target properties. e.g. atomic spin

The potential mixing constrained DFT method only requires a change to the residual before preconditioning

The density mixing constrained DFT method is a bit more involved.
"""

struct Constraint
    atom_pos          :: Vector{Float64}
    r_sm              :: Float64 #cutoff radius for atomic function
    r_cut             :: Float64 #smearing width for atomic function
    spin_target       :: Float64
    cons_resid_weight :: Float64
    λ                 :: Float64 #Lagrange multiplier for constraint
end

function Constraint(model::Model,idx::Int,spin_target::Float64,cons_resid_weight::Float64=0.01,r_sm::Float64=0.1)::Constraint
    atom_pos = model.positions[idx]
    psp = model.atoms[idx].psp
    r_cut = maximum(psp.rp)
    r_cut = max(r_cut,psp.rloc)
    return Constraint(atom_pos,r_sm,r_cut,spin_target,cons_resid_weight,0.0)
end

struct Constraints
    cons_vec    :: Vector{Constraint}
    overlap     :: Array{Float64,2} #overlap matrix for the different atomic functions
    overlap_inv :: Array{Float64,2}
end

function Constraints(cons_vec::Vector{Constraint})::Constraints
    overlap = calculate_overlap(cons_vec)
    overlap_inv = inv(overlap)
    return Constraints(cons_vec,overlap,overlap_inv)
end

function weight_fn(r::Vector{Float64},cons::Constraint)::Float64
    at_r = sqrt(sum((r - cons.atom_pos).^2))
    if at_r > cons.r_cut
        return 0.0
    elseif at_r < cons.r_cut-cons.r_sm
        return 1.0
    else
        x = (r_cut-at_r)/r_sm
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
    for (i,cons) in enumerat(cons_vec)
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
    
function integrate_atomic_functions(arr::Array{Float64,4},basis::PlaneWaveBasis,constraints::Constraints)::Vector{Float64}
    """
    integrate an array arr and the weight functions of each constraint
    """

    rvecs = collect(r_vectors(basis))

    spins = zeros(Float64,length(constraints)) #called spins since this is what you get for integrating the spin density

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

function orthogonalise_residual!(δV::Array{Float64,4},basis::PlaneWaveBasis,constraints::Constraints)::Array{Float64,3}
    """
    Orthogonalise the residual with respect to the atomic weight functions
        δV = δV -  ∑ᵢ wᵢ(r) ∑ⱼ (W)ᵢⱼ⁻¹∫δV(r')wⱼ(r')dr'
    """

    rvecs = collect(rvectors(basis))

    δV_w_i = integrate_atomic_functions(δV,basis,constraints)

    to_be_multiplied_by_weights = constraints.overlap_inv*δV_w_i

    for (i,cons) in enumerate(constraints.cons_vec)
        for j in eachindex(rvecs)
            δV[j,1] -=  weight_fn(rvecs[j],cons)*to_be_multiplied_by_weights#spin up potential
            δV[j,2] -= -weight_fn(rvecs[j],cons)*to_be_multiplied_by_weights#spin down potential
        end
        cons.λ = to_be_multiplied_by_weights
    end
end      

function add_resid_constraints!(δV::Array{Float64,4},dev_from_target::Vector{Float64},constraints::Constraints,basis::PlaneWaveBasis)
    """
    The part that's added to the residual is ∑ᵢ cᵢ wᵢ(r) ∑ⱼ(W)⁻¹ᵢⱼ (Nⱼ-Nⱼᵗ)
    """
    rvecs = collect(r_vectors(basis))
    W_ij = constraints.overlap_inv
    for (i,cons) in enumerate(constraints)
        factor = cons.cons_resid_weight*(W_ij[i,:]⋅dev_from_target)    
        for j in eachindex(rvecs)
            δV += weight_fn(r_vecs[j],cons)*factor
        end
    end
end

function add_constraint_to_residual!(δV::Array{Float64,4},ρ::Array{Float64,4}, basis::PlaneWaveBasis,constraints::Constraints)

    spins = integrate_atomic_functions(ρ[:,:,:,2],basis,constraints)

    spins -= [cons.spin_target for cons in constraints.cons_vec]

    orthogonalise_residual!(δV,basis,constraints)

    add_resid_contraints!(δV,spins,constraints)

end
