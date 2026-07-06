# BLADE

A GPU-accelerated radio astronomy DSP plugin for [CyberEther](https://cyberether.org).

The BLADE plugin (Breakthrough Listen Accelerated DSP Engine) is part of the [stelline.space](https://stelline.space) stack. It provides the signal-processing blocks of a radio telescope backend: beamforming, correlation, phasor generation, polarization conversion, detection, integration, and stacking. The CUDA kernels are compiled just-in-time at runtime for the exact tensor shapes flowing through the pipeline. The blocks were born as the beamforming engine of the [Allen Telescope Array](https://www.seti.org/ata) and are used in production there.

## Documentation

The full documentation lives at [stelline.space/docs](https://stelline.space/docs). It includes a reference page for every block: the Beamformer, the Phasor, the Correlator, the Polarizer, the Detector, the Integrator, and the Stacker.

## Example Flowgraphs

The `examples/` directory contains complete pipelines that are also bundled into the plugin package:

- Phasor generation from telescope metadata (`blade_phasor.yml`).
- Stacking and integration of a generated signal (`blade_stacker_integrator.yml`).

## Building

The plugin builds with [Meson](https://mesonbuild.com) and bundles its package as a CEP:

```bash
meson setup build
meson compile -C build blade_cep
```

The same artifact is produced by the container image: `docker build --target artifact --output .dist .`

## License

The BLADE plugin is distributed under the MIT License. See [LICENSE](LICENSE).

```
                           .-.
          .-""`""-.      |(@ @)
       _/`oOoOoOoOo`\_   \ \-/
      '.-=-=-=-=-=-=-.'   \/ \
        `-=.=-.-=.=-'      \ /\
           ^  ^  ^         _H_ \ art by jgs
```
