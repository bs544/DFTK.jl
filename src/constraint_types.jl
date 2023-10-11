"""
Define constraint structs that are used in both the Hamiltonian terms
These terms are needed when performing constrained DFT within the density mixing approach
"""

struct Constraint
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
    cons_resid_weight :: Float64
end

function Constraint(model::Model,idx::Int,cons_resid_weight::Float64=1.0,r_sm_frac::Float64=0.05;r_cut=nothing,target_spin=nothing,target_charge=nothing)::Constraint
    atom_pos = vector_red_to_cart(model,model.positions[idx])
    psp = model.atoms[idx].psp
    if isnothing(r_cut)
        r_cut = maximum(psp.rp)
        r_cut = max(r_cut,psp.rloc)
    end
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
    return Constraint(atom_pos,idx,spin,charge,r_sm,r_cut,target_spin,target_charge, cons_resid_weight)
end

mutable struct Constraints
    # make all the constraint information readily accessible, also precompute whatever you can to save on time

    cons_vec      :: Vector{Constraint} #kept if any specific information is needed
    overlap_charge:: Array{Float64,2}   #overlap matrix for the different atomic functions
    overlap_spin  :: Array{Float64,2}
    at_fn_arrays  :: Array{Array{Float64,3},2} # precomputed atomic functions
    at_fft_arrays :: Array{Array{ComplexF64,3},2} # precomputed atomic function fourier transforms
    res_wgt_arrs  :: Array{Float64,2}          # weights assigned to the lagrange multiplier updates
    lambdas       :: Array{Number,2}           # lagrange multipliers, setting as number for ForwardDiff
    is_constrained:: Array{Int64,2}            # mask for whether a constraint is applied
    target_values :: Array{Float64,2}          # target values, 0 if unconstrained
    current_values:: Array{Float64,2}          # current values, 0 if unconstrained
    dvol          :: Float64                   # just basis.dvol, useful when integrating
end

function Constraints(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Constraints
    atomic_fns = get_at_function_arrays(cons_vec,basis)
    at_fft_fns = map(x->fft(basis,x),atomic_fns)
    # atomic_derivs = get_deriv_at_fn_arrays(cons_vec,basis)
    overlap_charge,overlap_spin = calculate_overlap(atomic_fns,basis.dvol)

    res_wgt_arrs  = zeros(Float64,(length(cons_vec),2))
    lambdas       = zeros(Float64,(length(cons_vec),2))
    is_constrained= zeros(Float64,(length(cons_vec),2))
    target_values = zeros(Float64,(length(cons_vec),2))
    current_values= zeros(Float64,(length(cons_vec),2))

    for (i,cons) in enumerate(cons_vec)
        if cons.charge
            res_wgt_arrs[i,1]   = cons.cons_resid_weight
            is_constrained[i,1] = 1
            target_values[i,1]  = cons.target_charge
        end
        if cons.spin
            res_wgt_arrs[i,2]   = cons.cons_resid_weight
            is_constrained[i,2] = 1
            target_values[i,2]  = cons.target_spin
        end
    end

    return Constraints(cons_vec,overlap_charge,overlap_spin,atomic_fns,at_fft_fns,res_wgt_arrs,lambdas,is_constrained,target_values,current_values,basis.dvol)
end

function periodic_vect(r::AbstractVector,basis::PlaneWaveBasis)::AbstractVector
    """
    generate possible vectors based on unit cell translations (27 in total)
    return the one with the minimum distance
    there's likely a better way of doing this, but it only needs to be done once, so hopefully it's not too much bother
    """
    red_vector = vector_cart_to_red(basis.model,r)
    #the closest image will be in one of the 27 unit cells, so just focus on them
    disp = [1.0,0.0,-1.0]
    cell_disp = Array{Vector{Float64},3}(undef,3,3,3)
    for i=1:3; for j = 1:3; for k=1:3
        cell_disp[i,j,k] = [disp[i],disp[j],disp[k]]
    end;end;end
    red_vector_options = map(x->x+red_vector,cell_disp)
    red_vector_options = vector_red_to_cart.(basis.model,red_vector_options)
    red_vector_dists = norm.(red_vector_options)
    red_vector_idx = findmin(red_vector_dists)[2]
    return red_vector_options[red_vector_idx]
end

periodic_dist(r::AbstractVector{Float64},basis::PlaneWaveBasis) = norm(periodic_vect(r,basis))

function weight_fn(r::AbstractVector{Float64},cons::Constraint,basis::PlaneWaveBasis)::Float64
    at_r = periodic_dist(r-cons.atom_pos,basis)
    if at_r > cons.r_cut
        return 0.0
    elseif at_r < cons.r_cut-cons.r_sm
        return 1.0
    else
        x = (cons.r_cut-at_r)/cons.r_sm
        return x^2 * (3 + x*(1 + x*( -6 + 3*x)))
    end
end

function weight_fn_deriv(f::AbstractVector,cons::Constraint,basis::PlaneWaveBasis)::AbstractVector
    at_r = periodic_vect(r-cons.atom_pos,basis)
    dist = norm(at_r)
    at_r /= dist
    if dist > cons.r_cut || dist < cons.r_cut-cons.r_sm
        return 0.0
    else
        x = (cons.r_cut-dist)/cons.r_sm
        fact = -x*(6_x*(3_x*(-24+15*x)))/cons.r_sm
        return fact*at_r
    end
end     

function get_at_function_arrays(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Array{Array{Float64,3},2}
    """
    Calculate the weight functions just the once in an effort to speed things up.
    """
    r_vecs = collect(r_vectors_cart(basis))

    asize = size(r_vecs)
    cons_asize = (length(cons_vec),2)
    weights = [zeros(Float64,asize) for i=1:cons_asize[1], j=1:cons_asize[2]]
    for (i,cons) in enumerate(cons_vec)
        w_arr = zeros(Float64,asize)
        for j in eachindex(r_vecs)
            w_arr[j] = weight_fn(r_vecs[j],cons,basis)
        end
        if cons.charge
            weights[i,1] = w_arr
        end
        if cons.spin
            weights[i,2] = w_arr
        end
    end
    return weights
end

function get_deriv_at_fn_arrays(cons_vec::Vector{Constraint},basis::PlaneWaveBasis)::Array{Array{Number,3},2}
    """
    calculate the derivative of the atomic weights w.r.t position and store in an array of arrays
    """
    r_vecs = collect(r_vectors_cart(basis))

    asize = size(r_vecs)

    cons_asize = (1:length(cons_vec),1:2,1:3) #final dimension is for x,y and z derivatives

    weights = [zeros(Float64,asize) for (i,j,k) in Iterators.product(cons_asize...)]

    for (i,cons) in enumerate(cons_vec)
        dx_arr = zeros(Float64,asize)
        dy_arr = zeros(Float64,asize)
        dz_arr = zeros(Float64,asize)
        for j in eachindex(r_vecs)
            dx_arr[j], dy_arr[j], dz_arr[j] = weight_fn_deriv(r_vecs[j],cons,basis)
        end
        if cons.charge
            weights[i,1,:] = [dx_arr,dy_arr,dz_arr]
        end
        if cons.spin
            weights[i,1,:] = [dx_arr,dy_arr,dz_arr]
        end
    end

    return weights
end


function calculate_overlap(atom_fns::Array{Array{Float64,3},2},dvol::Float64)::Tuple{Array{Float64,2},Array{Float64,2}}
    """
    If the atomic functions have been committed to arrays, then the integration is a lot cheaper
    """
    n_constraint = size(atom_fns)[1]
    overlap_spin   = zeros(Float64,(n_constraint,n_constraint))
    overlap_charge = zeros(Float64,(n_constraint,n_constraint))
    for i = 1:n_constraint
        for j = i:n_constraint
            overlap_spin[i,j]   = sum(atom_fns[i,2].*atom_fns[j,2])
            overlap_charge[i,j] = sum(atom_fns[i,1].*atom_fns[j,1])

            overlap_spin[j,i]   = overlap_spin[i,j]
            overlap_charge[j,i] = overlap_charge[i,j]
        end
    end
    overlap_spin   .*= dvol
    overlap_charge .*= dvol

    return overlap_charge,overlap_spin

end

function integrate_atomic_functions(arr::Array{Float64,3},constraints::Constraints,idx::Int)::Vector{Float64}
    """
    Use the atomic functions to integrate an array arr
    """
    arr_values = Vector{Float64}(undef,length(constraints.cons_vec))
    for i = 1:length(constraints.cons_vec)
        arr_values[i] = sum(constraints.at_fn_arrays[i,idx].*arr)*constraints.dvol
    end
    return arr_values
end

function get_constraints(basis::PlaneWaveBasis)::Constraints
    for term in basis.terms
        if :constraints in fieldnames(typeof(term))
            return term.constraints
        end
    end
    return nothing
end

function update_constraints!(basis::PlaneWaveBasis,lambdas)
    for term in basis.terms
        if :constraints in fieldnames(typeof(term))
            term.constraints.lambdas = lambdas
        end
    end
end

function get_4d_at_fns(constraints::Constraints)
    """
    For each constraint get the representation of it in the spin up spin down representation,
    this is needed when applying χ0 to the atomic functions for the inner loop calculations
    """
    at_fn_arrs = Array{Array{Float64,4},2}(undef,size(constraints.at_fn_arrays))

    for i in CartesianIndices(constraints.at_fn_arrays)
        spin = i.I[2]==2
        factor = spin ? -1.0 : 1.0
        at_fn = constraints.at_fn_arrays[i]
        arr = zeros(Float64,size(at_fn)...,2)
        arr[:,:,:,1] = at_fn
        arr[:,:,:,2] = factor*at_fn
        at_fn_arrs[i] = arr
    end
    return at_fn_arrs
end


mutable struct ArrayAndConstraints
    """
    combines e.g. the density with the constraint information required for mixing:
        * Lagrange multiplier values
        * Weights for each constraint
    """
    arr     :: AbstractArray
    lambdas :: AbstractArray{Float64,2}
    weights :: AbstractArray{Float64,2}
end

ArrayAndConstraints(arr::AbstractArray,constraints::Constraints) = ArrayAndConstraints(arr,constraints.lambdas,constraints.res_wgt_arrs)

ArrayAndConstraints(arr::AbstractArray,basis::PlaneWaveBasis) = ArrayAndConstraints(arr,get_constraints(basis))

#overloading functions needed for the mixing of the density so that it includes the residual
Base.:+(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr + b.arr,a.lambdas+b.lambdas,a.weights)
Base.:-(a::ArrayAndConstraints,b::ArrayAndConstraints)= ArrayAndConstraints(a.arr - b.arr,a.lambdas-b.lambdas,a.weights)
Base.vec(a::ArrayAndConstraints) = vcat(vec(a.arr),vec(a.lambdas))# .* a.weights))
# Base.vec(a::ArrayAndConstraints) = vcat(vec(a.arr),vec(a.lambdas .* a.weights))
LinearAlgebra.norm(a::ArrayAndConstraints) = norm(a.arr) + norm(a.lambdas)# .* a.weights)
# LinearAlgebra.norm(a::ArrayAndConstraints) = norm(a.arr) + norm(a.lambdas .* a.weights)
Base.:*(a::Number,b::ArrayAndConstraints) = ArrayAndConstraints(a.*b.arr,a.*b.lambdas,b.weights)
Base.eltype(a::ArrayAndConstraints) = eltype(a.arr)
spin_density(a::ArrayAndConstraints) = spin_density(a.arr)
Base.copy(a::ArrayAndConstraints) = ArrayAndConstraints(a.arr,a.lambdas,a.weights)
# Base.size(a::ArrayAndConstraints) = size(a.arr)

#turns out you don't need to overload the broadcasting bits if you specify your type as an effective scalar with the following
Base.broadcastable(a::ArrayAndConstraints) = Ref(a) #this may cause problems later on, but works for now
# Base.broadcasted(::typeof(+),a::ArrayAndConstraints,b::ArrayAndConstraints) = ArrayAndConstraints(a.arr.+b.arr,a.lambdas.+b.lambdas,a.weights)
# Base.broadcasted(::typeof(-),a::ArrayAndConstraints,b::ArrayAndConstraints) = ArrayAndConstraints(a.arr.-b.arr,a.lambdas.-b.lambdas,a.weights)
# Base.broadcasted(::typeof(*),a::Number,b::ArrayAndConstraints) = ArrayAndConstraints(a.*b.arr,a.*b.lambdas,b.weights)


function back_to_array(x_new::Vector{Float64},x_old::ArrayAndConstraints)::ArrayAndConstraints
    """
    My attempt to crowbar in the ArrayAndConstraints into the anderson acceleration
    This is to replace the initial "reshape(xₙ₊₁, size(xₙ))"
    """
    arr_length = prod(size(x_old.arr))
    x_new_arr = reshape(x_new[begin:arr_length],size(x_old.arr))
    x_new_weight = x_old.weights
    x_new_λ = reshape(x_new[arr_length+1:end],size(x_old.lambdas))
    # x_new_λ ./= x_new_weight
    # x_new_λ .*= x_new_weight
    return ArrayAndConstraints(x_new_arr,x_new_λ,x_new_weight)
end

back_to_array(x_new::AbstractVector,x_old::AbstractArray) = reshape(x_new,size(x_old))

function get_density_deviation(ρout::Array,basis::PlaneWaveBasis)
    """
    Get the deviation of the output density from the constraint target values
    """
    constraints = get_constraints(basis)
    deriv_array = zeros(Float64,size(constraints.lambdas))

    spin_dens = nothing
    if length(size(ρout))==4 && size(ρout)[4]==2
        charge_dens = ρout[:,:,:,1]+ρout[:,:,:,2]
        spin_dens = ρout[:,:,:,1]-ρout[:,:,:,2]
    elseif length(size(ρout))==4
        charge_dens = ρout[:,:,:,1]
    else
        charge_dens = ρout
    end

    charges = integrate_atomic_functions(charge_dens,constraints,1)
    spins = !isnothing(spin_dens) ? integrate_atomic_functions(spin_dens,constraints,2) : zeros(Float64,size(charges))

    constraints.current_values[:,1] = charges
    constraints.current_values[:,2] = spins

    deriv_array = constraints.current_values-constraints.target_values

    deriv_array .*= constraints.is_constrained
    return deriv_array
end

function residual(ρout::Array,ρin_cons::ArrayAndConstraints,basis::PlaneWaveBasis;apply_overlap=true)::ArrayAndConstraints
    """
    Define the residual to include the gradient of the energy with respect to the lagrange multipliers
    """
    constraints = get_constraints(basis)

    deriv_array = get_density_deviation(ρout,basis)


    # this should be the only place that the weights are needed.
    # It acts effectively as a step size for the Lagrange multiplier update
    #getting this to be similar to the potential mixing approach, you need to multiply by the square root of the overlap matrix
    if apply_overlap
        S_c_inv_sqrt,S_s_inv_sqrt = overlap_inv_sqrt(constraints)
        deriv_array[:,1] = S_c_inv_sqrt*deriv_array[:,1]
        deriv_array[:,2] = S_s_inv_sqrt*deriv_array[:,2]
    end
    deriv_array .*= ρin_cons.weights 
    # deriv_array .*= basis.dvol

    weights  = constraints.res_wgt_arrs

    arr = ρout-ρin_cons.arr

    return ArrayAndConstraints(arr,deriv_array,weights)
end

function overlap_inv_sqrt(constraints)
    """
    We need the inverse square root of the overlaps, but only for the terms that are actually present, so shrink the matrices to the constrained elements and then expand back
    """
    n_constraints = length(constraints.cons_vec)
    chrg_const_eles = [i for i in 1:n_constraints if constraints.is_constrained[i,1]==1]
    spin_const_eles = [i for i in 1:n_constraints if constraints.is_constrained[i,2]==1]
    
    chrg_sub_mat = constraints.overlap_charge[chrg_const_eles,chrg_const_eles]
    spin_sub_mat = constraints.overlap_spin[spin_const_eles,spin_const_eles]

    chrg_sub_mat = sqrt(inv(chrg_sub_mat))
    spin_sub_mat = sqrt(inv(spin_sub_mat))

    larger_chrg_mat = zeros(Float64,n_constraints,n_constraints)
    larger_spin_mat = zeros(Float64,n_constraints,n_constraints)

    larger_chrg_mat[chrg_const_eles,chrg_const_eles] = chrg_sub_mat
    larger_spin_mat[spin_const_eles,spin_const_eles] = spin_sub_mat

    return larger_chrg_mat,larger_spin_mat   
end

# Add these utility functions, which are used in the inner loop and density mixing cDFT approaches
function lambdas_2_vector(lambdas,constraints::Constraints)
    """
    For the purposes of minimisation, the lambdas array needs to be a vector of only the elements that are being constrained
    """
    T = typeof(lambdas[1])
    T = T<:Int ? Number : T #generalise if T is an integer, since this can happen if lambdas[1] is 0
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
