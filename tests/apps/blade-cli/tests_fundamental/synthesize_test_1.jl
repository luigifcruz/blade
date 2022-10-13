include("helper.jl")

n_ant = 4
n_chan_perant = 128
n_time = 16384
n_pols = 2
n_bits = 8
n_beam = 1

# Zeros BFR5, with `1+0j` for cal_all
# Signal in [:, :, NCHAN/2, NANT/2]  of RAW data

recipe = createBeamformerRecipe(
	n_ant,
	n_chan_perant,
	n_time,
	n_pols,
	n_bits,
	n_beam
)
to_hdf5("/mnt/buf1/mydonsol_blade/basics/synthetic_test_1.bfr5", recipe)

header = createHeader(
	n_ant,
	n_chan_perant,
	n_time,
	n_pols,
	n_bits
)

data = Array(header) # [pol, time, chan, ant]
data .= div(typemax(real(eltype(data))), 16)
data[:, :, div(n_chan_perant,2), div(n_ant, 2)] .= div(typemax(real(eltype(data))), 2)

open("/mnt/buf1/mydonsol_blade/basics/synthetic_test_1.0000.raw", "w") do fio
	for i in 1:32
		write(fio, header)
		write(fio, data)
		header["PKTIDX"] += header["PIPERBLK"]
	end
end
