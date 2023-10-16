"""
Some initial functions to define the anderson mixing modification for the constrained DFT calculation with density mixing
"""
struct cDFTAndersonAcceleration
  m::Int                     # maximal history size
  iterates::Vector{Any}      # xₙ
  residuals::Vector{Any}     # Pf(xₙ)
  maxcond::Real              # Maximal condition number for Anderson matrix
  overlap_sqrt::Array{Float64,2}  # the square root of overlap matrix for the constraints
  is_constrained::Array{Int} # array saying which constraints are applied
end
cDFTAndersonAcceleration(constraints;m=10,maxcond=1e6) = cDFTAndersonAcceleration(m,[],[],maxcond,sqrt(get_overlap(constraints)),constraints.is_constrained)

function Base.push!(cDFT_anderson::cDFTAndersonAcceleration, xₙ, αₙ, Pfxₙ)
  push!(cDFT_anderson.iterates,  vec(xₙ))
  push!(cDFT_anderson.residuals, vec(Pfxₙ))
  if length(cDFT_anderson.iterates) > cDFT_anderson.m
      popfirst!(cDFT_anderson.iterates)
      popfirst!(cDFT_anderson.residuals)
  end
  @assert length(cDFT_anderson.iterates) <= cDFT_anderson.m
  @assert length(cDFT_anderson.iterates) == length(cDFT_anderson.residuals)
  cDFT_anderson
end

function (cDFT_anderson::cDFTAndersonAcceleration)(xₙ, αₙ, Pfxₙ) 
    """
    Does the actual mixing step. 
    Takes in: 
        the input density and Lagrange multiplier xₙ
        the mixing parameter αₙ
        the (preconditioned) residual Pfxₙ and the treated Lagrange multiplier gradients
    The inputs are of the type ArrayAndConstraints
    The Lagrange multiplier gradients are handled differently to the density residual due to the overlap stuff.
    When initially calculated, the Lagrange multipliers are initially multiplied by the inverse overlap matrix.
    For the inner products, a factor of the overlap matrix must be multiplied back in.

    The residuals will already have this overlap included, so the main thing to do differently to the regular 
    Anderson mixing is to modify the calculation of M to include this extra overlap term.
    """
  xs   = cDFT_anderson.iterates
  Pfxs = cDFT_anderson.residuals

  S_half = cDFT_anderson.overlap_sqrt

  # Special cases with fast exit
  cDFT_anderson.m == 0 && return xₙ .+ αₙ .* Pfxₙ
  if isempty(xs)
      push!(cDFT_anderson, xₙ, αₙ, Pfxₙ)
      return xₙ .+ αₙ .* Pfxₙ
  end

  n_Lagrange = length(xs[1])-prod(size(xₙ.arr))
  M = hcat(Pfxs...) .- vec(Pfxₙ) # Mᵢⱼ = (Pfxⱼ)ᵢ - (Pfxₙ)ᵢ
  # M = apply_overlap(M,S_half,n_Lagrange)
  # We need to solve 0 = M' Pfxₙ + M'M βs <=> βs = - (M'M)⁻¹ M' Pfxₙ

  # Ensure the condition number of M stays below maxcond, else prune the history
  Mfac = qr(M)
  while size(M, 2) > 1 && cond(Mfac.R) > cDFT_anderson.maxcond
      M = M[:, 2:end]  # Drop oldest entry in history
      popfirst!(cDFT_anderson.iterates)
      popfirst!(cDFT_anderson.residuals)
      Mfac = qr(M)
  end

  xₙ₊₁ = vec(xₙ) .+ αₙ .* vec(Pfxₙ)
  βs   = -(Mfac \ apply_overlap(vec(Pfxₙ),S_half,n_Lagrange))
  βs = to_cpu(βs)  # GPU computation only : get βs back on the CPU so we can iterate through it
  for (iβ, β) in enumerate(βs)
      xₙ₊₁ .+= β .* (xs[iβ] .- vec(xₙ) .+ αₙ .* (Pfxs[iβ] .- vec(Pfxₙ)))
  end

  push!(cDFT_anderson, xₙ, αₙ, Pfxₙ)
  back_to_array(xₙ₊₁,xₙ) # reshape(xₙ₊₁, size(xₙ))
end