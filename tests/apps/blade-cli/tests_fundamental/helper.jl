using Blio: GuppiRaw
using BeamformerRecipes
using Printf


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
	telInfo.latitude = (34*60 + 04) * 60 + 43.0
	telInfo.longitude = (-107*60 + 37)*60 + 4.0
	telInfo.altitude = 2124
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