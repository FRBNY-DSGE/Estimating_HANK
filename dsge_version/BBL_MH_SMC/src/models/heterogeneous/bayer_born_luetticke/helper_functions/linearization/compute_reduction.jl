

function compute_rediction(m::BayerBornLuetticke)

SCov = zeros(1080, 1080)
for i in m.exogenous_shocks
    SCov[] = get_setting(m, Symbol("σ_", i)).^2
end

#--------------------------------------------------
#STEP 2: produce eigenvalue decomposition
#--------------------------------------------------
compression_indices = get_setting(m, :dct_compression_indices)
nstates = get_setting(m, :n_backward_looking_states)
ntotal = size(compression_indices[:Vm]) + size(compression_indices[:Vk]) + size(compression_indices[:copula])

Dindex = compression_indices[:copula]
evalS, evecS = eigen(StateCOVAR[Dindex, Dindex])
keepD = abs.(evalS).>maximum(evalS)*get_setting(m, :further_compress_critS)
indKeepD = Dindex[keepD]
nstates_reduced = nstates - length(Dindex) + length(indKeepD)

Vindex = [compression_indices[:Vm] ; compression_indices[:Vk]]
evalC, evecC = eigen(ControlCOVAR[Vindex .- nstates, Vindex .- nstates])
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
PRightALl_aux[Dindex, Dindex] = evecS
PRightAll_aux[Vindex, Vindex] = evecC
keep = ones(Bool, ntotal)
keep[Dindex[.!keepD]] .= false
keep[Vindex[.!keepV]] .= false
m <= Setting(:PrightAll, PRightAll_aux[:, keep])

update_compression_indices!(m, [:Vm, :Vk, :copula], keepV[keepV][1:2], keepV[keepV][3:end], keepD[keepD])

end
