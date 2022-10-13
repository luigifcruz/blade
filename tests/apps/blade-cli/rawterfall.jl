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
    data = (data[1, :, :, :].^2) + (data[2, :, :, :].^2) # real [time, chan, ant]
    data = (sum(data, dims=3))/(size(data, 3)) # real [time, chan, 1]

    fig = heatmap(data[:, :, 1], title=stem)
    xlabel!(fig, "Frequency Channels")
    ylabel!(fig, "Time samples")
    
    savefig(fig, @sprintf("%s.png", stem))
  close(fio)
end
