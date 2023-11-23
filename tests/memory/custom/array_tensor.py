import blade as bl

if __name__ == "__main__":

    host_input = bl.array_tensor((1, 2, 3, 4), dtype=bl.f32, device=bl.cpu)

    print(host_input)

    assert host_input.unified == False
    assert host_input.hash != 0

    assert host_input.shape.number_of_aspects == 1
    assert host_input.shape.number_of_frequency_channels == 2
    assert host_input.shape.number_of_time_samples == 3
    assert host_input.shape.number_of_polarizations == 4