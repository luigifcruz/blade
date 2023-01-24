using Blio: GuppiRaw
using BeamformerRecipes
using Printf
using Plots

ENV["GKSwstype"]="nul" # disable plots display

function createHeader(
	n_ant::Integer,
	n_chan_perant::Integer,
	n_time::Integer,
	n_pols::Integer,
	n_bits::Integer
)::GuppiRaw.Header
	header = GuppiRaw.Header()
	header["BLOCSIZE"]	= div((n_ant*n_chan_perant*n_time*n_pols*2*n_bits), 8)
	header["NANTS"]			= n_ant
	header["OBSNCHAN"]	= n_ant*n_chan_perant
	header["NPOL"]			= n_pols
	header["NBITS"]			= n_bits
	header["DIRECTIO"]	= 0
	header["SCHAN"]			= 0
	header["CHAN_BW"]		= 0.5
	header["OBSBW"]			= 0.5*n_chan_perant
	header["TBIN"]			= 1.0/(header["CHAN_BW"]*1e6)
	header["OBSFREQ"]		= 1500
	header["SYNCTIME"]	= 0
	header["PIPERBLK"]	= n_time
	header["PKTIDX"]		= 0
	header["DUT1"]			= 0.0
	header["AZ"]				= 0.0
	header["EL"]				= 0.0
	header["SRC_NAME"]	= "SYNTHESIZED"
	header["TELESCOP"]	= "FAKE"
	return header
end

function createBeamformerRecipe(
	n_ant::Integer,
	n_chan_perant::Integer,
	n_time::Integer,
	n_pols::Integer,
	n_bits::Integer,
	n_beam::Integer
)::BeamformerRecipe

	dimInfo = DimInfo()
	dimInfo.nants = n_ant
	dimInfo.nchan = n_chan_perant
	dimInfo.ntimes = n_time
	dimInfo.npol = n_pols
	dimInfo.nbeams = n_beam

	beamInfo = BeamInfo()
	beamInfo.ras = zeros(Float64, (dimInfo.nbeams))
	beamInfo.decs = zeros(Float64, (dimInfo.nbeams))
	beamInfo.src_names = repeat(["X"], dimInfo.nbeams)

	calInfo = CalInfo()
	calInfo.refant = "ant00"
	calInfo.cal_K = zeros(Float32, (dimInfo.nants, dimInfo.npol))
	calInfo.cal_B = zeros(ComplexF32, (dimInfo.nants, dimInfo.npol, dimInfo.nchan))
	calInfo.cal_G = zeros(ComplexF32, (dimInfo.nants, dimInfo.npol))
	calInfo.cal_all = ones(ComplexF32, (dimInfo.nants, dimInfo.npol, dimInfo.nchan))

	delayInfo = DelayInfo()
	delayInfo.delays = zeros(Float64, (dimInfo.nants, dimInfo.nbeams, 1))
	delayInfo.rates = zeros(Float64, (dimInfo.nants, dimInfo.nbeams, 1))
	delayInfo.time_array = zeros(Float64, (1))
	delayInfo.jds = zeros(Float64, (1))
	delayInfo.dut1 = 0.0

	obsInfo = ObsInfo()
	obsInfo.obsid = "SYNTHETIC"
	obsInfo.freq_array = zeros(Float64, (dimInfo.nchan))
	obsInfo.phase_center_ra = beamInfo.ras[1]
	obsInfo.phase_center_dec = beamInfo.decs[1]
	obsInfo.instrument_name = "Unknown"

	telInfo = TelInfo()
	telInfo.antenna_positions = zeros(Float64, (3, dimInfo.nants))
	telInfo.antenna_position_frame = "xyz"
	telInfo.antenna_names = [@sprintf("ant%03d", i) for i in 0:dimInfo.nants-1]
	telInfo.antenna_numbers = collect(0:dimInfo.nants-1)
	telInfo.antenna_diameters = collect(6 for i in 1:dimInfo.nants)
	telInfo.latitude = 34.07881373419933
	telInfo.longitude = -107.61833419041476 
	telInfo.altitude = 2114.8787108091637
	telInfo.telescope_name = "Unknown"

	BeamformerRecipe(
		dimInfo,
		telInfo,
		obsInfo,
		calInfo,
		beamInfo,
		delayInfo
	)
end

function polMagnitude(data::Array, dims::Integer)
	reshape(
		sqrt.(sum(abs.(data) .^ 2, dims=dims)),
		[dim for (dimidx, dim) in enumerate(size(data)) if dimidx != dims]...
	)
end

# function polAngle(data, dims)
# 	sqrt.(sum(abs.(data) .^ 2), dims=dims)
# end

function saveInputPlot(
	rawdata::Array, # complex [pol, time, chan, ant]
	calcoeff::Array,# complex [ant, pol, chan]
	title::String,
	directory="."
)
	
	nants = size(rawdata, 4)
	plotdata = polMagnitude(rawdata, 1)
	minval, maxval = min(plotdata...), max(plotdata...)

	l = @layout [grid(2,Integer(ceil(nants/2))) a{0.05w}]

	plot(
		[
			plot(
				heatmap(plotdata[:, :, ant_i], ylabel="Time Samples", colorbar=false, clims=(minval, maxval)),
				plot(polMagnitude(calcoeff[ant_i, :, :], 1), xlabel="Frequency Channels", ylabel="Magnitude"),
				layout=grid(2,1,heights=[0.8, 0.2]),
				size=(1200,1000),
				link=:x
			)
			for ant_i in 1:nants
		]...,
		heatmap(LinRange(minval, maxval, 101).*ones(101,1), legend=:none, xticks=:none, yticks=(1:10:101, string.(0:0.1:1))),
		layout = l,
		plot_title = title
	)
	savefig(joinpath(directory, @sprintf("%s.png", title)))
end

function generateTestInputs(
	title::String,
	bfr5callback::Function,
	rawcallback::Function;
	
	directory::String,
	iterateRawcallback::Bool = false,
	
	n_ant::Integer = 4,
	n_chan_perant::Integer = 128,
	n_time::Integer = 16384,
	n_pols::Integer = 2,
	n_bits::Integer = 8,
	n_beam::Integer = 1,

	n_blocks::Integer = 32,
)

	recipe = createBeamformerRecipe(
		n_ant,
		n_chan_perant,
		n_time,
		n_pols,
		n_bits,
		n_beam
	)
	bfr5callback(recipe)

	header = createHeader(
		n_ant,
		n_chan_perant,
		n_time,
		n_pols,
		n_bits
	)

	data = Array(header) # [pol, time, chan, ant]
	rawcallback(header, data)

	# plot input files
	saveInputPlot(
		data,
		recipe.calinfo.cal_all,
		title,
		"./plots"
	)

	# output input files
	stempath = joinpath(directory, title)
	to_hdf5(@sprintf("%s.bfr5", stempath), recipe)
	open(@sprintf("%s.0000.raw", stempath), "w") do fio
		for i in 1:n_blocks
			write(fio, header)
			if iterateRawcallback && i > 1
				rawcallback(header, data)
			end
			write(fio, data)
			header["PKTIDX"] += header["PIPERBLK"]
		end
	end

end
