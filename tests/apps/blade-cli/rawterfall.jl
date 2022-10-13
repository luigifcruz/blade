using Printf
using Plots
using Blio: GuppiRaw

ENV["GKSwstype"]="nul" # disable plots display

for filestem in ARGS

  grheader = GuppiRaw.Header()
  fio = open(filestem, "r")
    stem = split(filestem, "/")[end]

    read!(fio, grheader)
    println(grheader)
    data = Array(grheader)
    read!(fio, data) # complex [pol, time, chan, ant]
    data = abs.(data) # real [pol, time, chan, ant]
    data = sqrt.((data[1, :, :, :].^2) + (data[2, :, :, :].^2)) # real [time, chan, ant]

    nants = size(data, 3)

    ant_plots = Vector(undef, nants)
    for ant_i in 1:nants
      ant_plots[ant_i]= heatmap(data[:, :, ant_i])
    end
    
    plot(ant_plots..., layout=nants, title=stem, colorbar = true)
    xlabel!("Frequency Channels")
    ylabel!("Time samples")
    savefig(@sprintf("%s.png", stem))
  close(fio)
end
