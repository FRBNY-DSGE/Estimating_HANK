function init_pseudo_observable_mappings!(m::BayerBornLuetticke)

    pseudo_names = []

    # Create PseudoObservable objects
    pseudo = OrderedDict{Symbol,PseudoObservable}()
    for k in pseudo_names
        pseudo[k] = PseudoObservable(k)
    end

    # Add to model object
    m.pseudo_observable_mappings = pseudo
end
