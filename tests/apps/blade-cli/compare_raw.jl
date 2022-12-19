using Printf
using FFTW
using Blio: GuppiRaw # https://github.com/MydonSolutions/Blio.jl

struct Result
  message::Union{String, Nothing}
  value::Bool
end

function mapToFloat(value::Integer, type::Type)
  return value < 0 ? -1.0*(value / typemin(type)) : value / typemax(type)
end

function mapToFloat(value::Complex{<:Integer}, type::Type)
  return complex(mapToFloat(real(value), real(type)), mapToFloat(imag(value), real(type)))
end

function compare(i_data, o_data, atol=0.01)::Result
  correct_count = 0
  count = 0
  length_ratio = length(i_data) / length(o_data)
  time_ratio = size(i_data)[2] / size(o_data)[2]
  if length_ratio != 1 && !(length_ratio == time_ratio )
    return Result(@sprintf("Shape mismatch: %s != %s", size(i_data), size(o_data)), false)
  end
  dims = length(i_data) >= length(o_data) ? size(i_data) : size(o_data)

  for i in CartesianIndices(size(i_data)[2:end])
    dims_correct = all(isapprox.(real(i_data[:, i]), real(o_data[:, i]), atol=atol)) && all(isapprox.(imag(i_data[:, i]), imag(o_data[:, i]), atol=atol))
    if !dims_correct
      if count - correct_count < 100
        println(@sprintf("Polarization data mismatch @ %s: %s != %s\n\t(diff: %s)", i, i_data[:, i], o_data[:, i], i_data[:, i] - o_data[:, i]))
      elseif count - correct_count == 100
        println("... That's 100 mismatch-printouts, omitting the rest.")
      end
    else
      correct_count += 1
    end
    count += 1
  end

  Result(@sprintf("%03.06f%% correct (%d/%d)", correct_count/count*100, correct_count, count), correct_count == count)
end

i_grheader = GuppiRaw.Header()
o_grheader = GuppiRaw.Header()

i_fio = open(ARGS[1], "r")
o_fio = open(ARGS[2], "r")

  read!(i_fio, i_grheader)
  i_data = Array(i_grheader)
  read!(i_fio, i_data)
	if eltype(i_data) <: Complex{<:Integer}
  	i_data = mapToFloat.(i_data, eltype(i_data))
	end

  read!(o_fio, o_grheader)
  o_data = Array(o_grheader)
  read!(o_fio, o_data)

  atol = 0.001

  println("\n", compare(i_data, o_data, atol))

close(i_fio)
close(o_fio)
