import blade as bl

if __name__ == "__main__":

    host_input = bl.phasor_tensor((1, 2, 3, 4, 5), dtype=bl.f32, device=bl.cpu)

    print(host_input)

    assert host_input.unified == False
    assert host_input.hash != 0

    assert host_input.shape.number_of_beams == 1
    assert host_input.shape.number_of_antennas == 2
    assert host_input.shape.number_of_frequency_channels == 3
    assert host_input.shape.number_of_time_samples == 4
    assert host_input.shape.number_of_polarizations == 5