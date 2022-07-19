using BeamformerRecipes
using Random
using Printf

# Remember, Julia's dimensions are fastest...slowest

Random.seed!(0)

dimInfo = DimInfo()
dimInfo.nants = 20
dimInfo.npol = 2
dimInfo.nchan = 128
dimInfo.nbeams = 8
dimInfo.ntimes = 16384

beamInfo = BeamInfo()
beamInfo.ras = rand(Float64, (dimInfo.nbeams))
beamInfo.decs = rand(Float64, (dimInfo.nbeams))
beamInfo.src_names = repeat(["X"], dimInfo.nbeams)

calInfo = CalInfo()
calInfo.refant = "ant00"
calInfo.cal_K = rand(Float32, (dimInfo.nants, dimInfo.npol))
calInfo.cal_B = rand(ComplexF32, (dimInfo.nants, dimInfo.npol, dimInfo.nchan))
calInfo.cal_G = rand(ComplexF32, (dimInfo.nants, dimInfo.npol))
calInfo.cal_all = rand(ComplexF32, (dimInfo.nants, dimInfo.npol, dimInfo.nchan))

delayInfo = DelayInfo()
delayInfo.delays = rand(Float64, (dimInfo.nants, dimInfo.nbeams, dimInfo.ntimes))
delayInfo.rates = rand(Float64, (dimInfo.nants, dimInfo.nbeams, dimInfo.ntimes))
delayInfo.time_array = rand(Float64, (dimInfo.ntimes))
delayInfo.jds = rand(Float64, (dimInfo.ntimes))
delayInfo.dut1 = 0.0

obsInfo = ObsInfo()
obsInfo.obsid = "SYNTHETIC"
obsInfo.freq_array = rand(Float64, (dimInfo.nchan))
obsInfo.phase_center_ra = beamInfo.ras[1]
obsInfo.phase_center_dec = beamInfo.decs[1]
obsInfo.instrument_name = "Unknown"

telInfo = TelInfo()
telInfo.antenna_positions = rand(Float64, (3, dimInfo.nants))
telInfo.antenna_position_frame = "xyz"
telInfo.antenna_names = [@sprintf("ant%03d", i) for i in 0:dimInfo.nants-1]
telInfo.antenna_numbers = collect(0:dimInfo.nants-1)
telInfo.antenna_diameters = collect(6 for i in 1:dimInfo.nants)
telInfo.latitude = (34*60 + 04) * 60 + 43.0
telInfo.longitude = (-107*60 + 37)*60 + 4.0
telInfo.altitude = 2124
telInfo.telescope_name = "Unknown"

recipe = BeamformerRecipe(
	dimInfo,
	telInfo,
	obsInfo,
	calInfo,
	beamInfo,
	delayInfo
)

to_hdf5("./synthesized_input.bfr5", recipe)