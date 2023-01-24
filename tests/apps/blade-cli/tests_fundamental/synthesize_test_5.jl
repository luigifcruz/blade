include("helper.jl")

# Zeros BFR5, with `1+0j` for cal_all
# Complex-Exponential Signal repeated throughout RAW data

generateTestInputs(
	"synthetic_test_5",
	function (recipe)
		recipe
	end,
	function (header, data)
		n_ant = header["NANTS"]
		n_chan_perant = div(header["OBSNCHAN"], n_ant)
		n_time = header["PIPERBLK"]
		n_pol = header["NPOL"]
		time_0 = header["PKTIDX"]
		sample_component_type = real(eltype(data))
		type_max = typemax(sample_component_type)

		time_samples = round.(exp.(((0:n_time-1) .+ time_0) .* im*2*pi*1e-3) .* div(type_max, 2))

		for poli in 1:n_pol
			for anti in 1:n_ant
				for chani in 1:n_chan_perant
					data[poli,:,chani,anti] .= time_samples;
				end
			end
		end

	end,
	directory = "/mnt/buf1/mydonsol_blade/basics",
	iterateRawcallback = true
)
