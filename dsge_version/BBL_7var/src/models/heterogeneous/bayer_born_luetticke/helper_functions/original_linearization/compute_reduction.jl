function compute_reduction(m::BayerBornLuetticke)  #may also import theta
#for now, hard coded are: nstates, the inputs for LOMstate and State2Control,
## nm, nk here are 40 whereas in BBL are 50 each, different lengths of value function compression indices
#id = get_setting(m,:prime_and_noprime_indices)::OrderedDict{Symbol, UnitRange{Int}} #double check that this prints indices,otherwise call construct
θ = parameters2namedtuple(m)
id = construct_prime_and_noprime_indices(m; only_aggregate = false) #perhaps pass it in/set it inside the jacobian file so that don't have to call it again

id_keys = keys(id)
id_values = values(id)
#tNo = get_setting(m, :nk) + get_setting(m, :ny) + get_setting(m, :nm)
#tNo4 = length(get_setting(m, :dct_compression_indices)[:copula])
shocks = keys(m.exogenous_shocks)
#put this into bbl initialization eventually
shock2state_name_map = get_setting(m,:shock2state)
#shock2state_name_map = Dict(:A_sh => :A_t,:Z_sh => :Z_t,:Ψ_sh => :Ψ_t,:μ_p_sh => :μ_p_t,:μ_w_sh => :μ_w_t, :G_sh => :G_sh_t,:P_sh => :P_sh_t,:R_sh => :R_sh_t,:S_sh => :S_sh_t)

shock_index = Dict()

## shock index alignment is off (for BBL is not necessarily linearly asigned from 1080 onwards)
for i in shocks
   shock_index[i] = id[shock2state_name_map[i]] ## parallels BBL getfield(sr.indexes,i) in their compute_reduction.jl
end
#println(shock_index)

#-------------------------------------------------
#STEP 1: Long Run Covariance
#-------------------------------------------------

sd_dictionary = get_setting(m, :shock_to_deviation_dict)
nstates = get_setting(m,:n_backward_looking_states)
ntotal = get_setting(m,:n_model_states)
SCov = zeros(nstates, nstates)
for i in shocks

    SCov[shock_index[i], shock_index[i]] .=(θ[sd_dictionary[i]]).^2
end
@save "scov.jld2" SCov

StateCOVAR = lyapd(get_setting(m, :LOMstate), SCov)

State2Control = get_setting(m, :State2Control)
ControlCOVAR = State2Control*StateCOVAR*State2Control'
ControlCOVAR = (ControlCOVAR + ControlCOVAR') ./ 2

#--------------------------------------------------
#STEP 2: produce eigenvalue decomposition
#--------------------------------------------------

#compression_indices = get_setting(m, :dct_compression_indices)
#nstates = get_setting(m, :n_backward_looking_states)
#ntotal = length(compression_indices[:Vm]) + length(compression_indices[:Vk]) + length(compression_indices[:copula])

Dindex = id[:copula_t] ## values of the indices do not match
evalS, evecS = eigen(StateCOVAR[Dindex, Dindex])


keepD = abs.(evalS).>maximum(evalS)*get_setting(m, :further_compress_critS)

indKeepD = Dindex[keepD]
nstates_reduced = nstates - length(Dindex) + length(indKeepD)

#may not want to be using the literal compression indices bc are different than the regular indices variable
Vindex = [id[:Vm_t] ; id[:Vk_t]]
#Vindex = 1081:2119
evalC, evecC = eigen(ControlCOVAR[Vindex.-nstates, Vindex.-nstates])
keepV = abs.(evalC).>maximum(evalC)*get_setting(m, :further_compress_critC)
indKeepV = Vindex[keepV]

#-----------------------------------------------------------
#Setp 3: Put together projection matrices and update indexes
#-----------------------------------------------------------

PRightStates_aux = float(I[1:nstates, 1:nstates])
PRightStates_aux[Dindex, Dindex] = evecS
keep = ones(Bool, nstates)
keep[Dindex[.!keepD]] .= false
m <= Setting(:PRightStates, PRightStates_aux[:, keep])

PRightAll_aux = float(I[1:ntotal, 1:ntotal])
PRightAll_aux[Dindex, Dindex] = evecS
PRightAll_aux[Vindex, Vindex] = evecC
keep = ones(Bool, ntotal)
keep[Dindex[.!keepD]] .= false
keep[Vindex[.!keepV]] .= false

PRightAll = PRightAll_aux[:,keep]
m <= Setting(:PRightAll, PRightAll_aux[:, keep])

#update_compression_indices!(m, [:Vm, :Vk, :copula], keepV[keepV][1:2], keepV[keepV][3:end], keepD[keepD])
# seems like more about
id_reduced = Dict()
for key in id_keys
    id_reduced[key] = id[key]
end
id_reduced[:copula_t] = Int.(indKeepD)
id_reduced[:Vm] = Int.(indKeepV[indKeepV.<= last(id[:Vm_t])])
id_reduced[:Vk] = Int.(indKeepV[indKeepV.> last(id[:Vm_t])])

update_compression_indices!(m, [:Vm, :Vk, :copula], id_reduced[:Vm], id_reduced[:Vk], id_reduced[:copula_t])
setup_indices!(m)
#m <= Setting(:linearize_heterogeneous_block, false) #once reduced, only want the aggregate jacobian to be computed
m[:A] = PRightAll' * m[:A].value * PRightAll
m[:B] = PRightAll' * m[:B].value * PRightAll

#return id_reduced #may need to save this separately and the ordering of this may need to change, look to setup_indices and construct_prime_and_noprime for further guidance
end

# create a function here to generate reduced indices and variables
