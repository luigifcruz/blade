include("helper.jl")
using Dates

using ATA_BFR5_Genie # https://github.com/MydonSolutions/ata_bfr5_genie

# Zeros BFR5, with `1+0j` for cal_all
# Complex-Exponential Signal repeated throughout RAW data

now_unix = floor(datetime2unix(now()))
n_blocks = 32
observation_frequency_center_Mhz = 1500
channel_bandwidth_Mhz = 0.5

generateTestInputs(
	"synthetic_test_rand",
	function (recipe)
		n_ant = recipe.diminfo.nants
		n_beam = recipe.diminfo.nbeams
		n_chan_perant = recipe.diminfo.nchan
		n_time = recipe.diminfo.ntimes
		
		recipe.telinfo.antenna_positions = rand(Float64, (3, recipe.diminfo.nants)) .* 10000.0
		
		recipe.beaminfo.ras = rand(Float64, (recipe.diminfo.nbeams)) .* 0.01
		recipe.beaminfo.decs = rand(Float64, (recipe.diminfo.nbeams)) .* 0.01

		chan0_Mhz = observation_frequency_center_Mhz - (recipe.diminfo.nchan/2)*channel_bandwidth_Mhz
		recipe.obsinfo.freq_array = collect(0:recipe.diminfo.nchan-1) .* channel_bandwidth_Mhz*1e6 .+ chan0_Mhz*1e6
		recipe.obsinfo.freq_array /= 1e9 # stored in GHz
		recipe.obsinfo.phase_center_ra = recipe.beaminfo.ras[1] - rand(Float64)*0.01
		recipe.obsinfo.phase_center_dec = recipe.beaminfo.decs[1] - rand(Float64)*0.01

		recipe.delayinfo.time_array = collect(0:n_blocks-1)
		recipe.delayinfo.time_array .*= n_time
		recipe.delayinfo.time_array .+= floor(0.5 * n_time)
		recipe.delayinfo.time_array ./= 1e6*channel_bandwidth_Mhz
		recipe.delayinfo.time_array .+= fill(now_unix, (n_blocks))

		recipe.delayinfo.jds = [
			(unix / 86400) + 2440587.5
			for unix in recipe.delayinfo.time_array
		]
		recipe.calinfo.cal_all = rand(ComplexF32, (recipe.diminfo.nants, recipe.diminfo.npol, recipe.diminfo.nchan))

		recipe.delayinfo.delays = cat(
			(
				ATA_BFR5_Genie.calculateBeamDelays(
					recipe.telinfo.antenna_positions,
					1,
					recipe.obsinfo.phase_center_ra, recipe.obsinfo.phase_center_dec,
					hcat(recipe.beaminfo.ras, recipe.beaminfo.decs)',
					recipe.telinfo.longitude, recipe.telinfo.latitude, recipe.telinfo.altitude,
					unix, recipe.delayinfo.dut1
				)
				for unix in recipe.delayinfo.time_array
			)...
			;
			dims=3
		)

	end,
	function (header, data)
		n_ant = header["NANTS"]
		n_chan_perant = div(header["OBSNCHAN"], n_ant)
		n_time = header["PIPERBLK"]
		n_pol = header["NPOL"]
		
		header["SYNCTIME"] 	= now_unix
		header["OBSFREQ"] 	= observation_frequency_center_Mhz
		header["CHAN_BW"] 	= channel_bandwidth_Mhz
		header["OBSBW"]			= channel_bandwidth_Mhz*n_chan_perant
		header["TBIN"]			= 1.0/(header["CHAN_BW"]*1e6)

		data .= rand(eltype(data), (n_pol, n_time, n_chan_perant, n_ant))
	end,
	directory = "/mnt/buf1/mydonsol_blade/basics",
	iterateRawcallback = true,
	n_blocks = n_blocks
)
