include("helper.jl")

# Zeros BFR5, with `1+0j` for cal_all
# Signal in [:, :, NCHAN/2, NANT/2]  of RAW data

generateTestInputs(
	"synthetic_test_2",
	function (recipe)
		n_ant = recipe.diminfo.nants
		n_chan_perant = recipe.diminfo.nchan
		recipe.calinfo.cal_all .= 1/16
		recipe.calinfo.cal_all[div(n_ant, 2), :, div(n_chan_perant,2)] .= 1
	end,
	function (header, data)
		n_ant = header["NANTS"]
		n_chan_perant = div(header["OBSNCHAN"], n_ant)
		background = div(typemax(real(eltype(data))), 16)
		foreground = div(typemax(real(eltype(data))), 2)

		data .= background
		data[:, :, div(3*n_chan_perant,5), :] .= foreground

		data[:, :, div(n_chan_perant,2), div(n_ant, 2)] .= foreground
	end,
	directory = "/mnt/buf1/mydonsol_blade/basics"
)
