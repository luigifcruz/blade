include("helper.jl")

# Zeros BFR5, with `1+0j` for cal_all
# Signal in [:, :, NCHAN/2, NANT/2]  of RAW data

generateTestInputs(
	"synthetic_test_1",
	function (recipe)
		recipe
	end,
	function (header, data)
		n_ant = header["NANTS"]
		n_chan_perant = div(header["OBSNCHAN"], n_ant)
		data .= div(typemax(real(eltype(data))), 16)
		data[:, :, div(n_chan_perant,2), div(n_ant, 2)] .= div(typemax(real(eltype(data))), 2)
	end,
	directory = "/mnt/buf1/mydonsol_blade/basics"
)
