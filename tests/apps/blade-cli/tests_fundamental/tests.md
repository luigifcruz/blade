# Fundamental Beamforming Tests

For the purposes of brevity, the RAW data is spoken about programmatically, with dimensions [pol, time, chan, ant].


<details><summary>0. cal_all Ones, delays Zeros, RAW signal in [:, :, NCHAN/2, :]</summary>


<details><summary>GUPPI RAW Input</summary>

![synthetic_test_0.0000.raw](./plots/synthetic_test_0.0000.raw.png)

</details>

<details><summary>Beamformed Output (No upchannelization)</summary>

![synthetic_test_0_c1_beam0](./plots/synthetic_test_0_c1_beam0.png)
</details>

<details><summary>Beamformed Output (upchannelization rate of 4)</summary>

![synthetic_test_0_c4_beam0](./plots/synthetic_test_0_c4_beam0.png)
</details>


</details>


<details><summary>1. cal_all Ones, delays Zeros, RAW signal in [:, :, NCHAN/2, NANT/2]</summary>


<details><summary>GUPPI RAW Input</summary>

![synthetic_test_1.0000.raw](./plots/synthetic_test_1.0000.raw.png)

</details>

<details><summary>Beamformed Output (No upchannelization)</summary>

![synthetic_test_1_c1_beam0](./plots/synthetic_test_1_c1_beam0.png)
</details>

<details><summary>Beamformed Output (upchannelization rate of 4)</summary>

![synthetic_test_1_c4_beam0](./plots/synthetic_test_1_c4_beam0.png)
</details>


</details>