#include "blade/base.hh"
#include "blade/logger.hh"
#include "blade/runner.hh"
#include "blade/pipelines/vla/mode_b.hh"

extern "C" {
#include "guppiraw.h"
}

#include <unistd.h>
#include <time.h>

using namespace Blade;
using namespace Blade::Pipelines::VLA;

void usage(const char *argv0) {
  printf(
    "Usage: %s [options...] FILEPATH\n"
    "\n"
    "Options:\n"
    "  -b             Number of beams to form [1]\n"
    "  -c             Coarse-channel rate [1]\n"
    "  -t             Fine-time rate [32]\n"
    "  -u             Up-channelization rate [1]\n"
    "\n"
    "  -h             Show this message\n"
    , argv0
  );
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    return 1;
  }

  U64 opt_channelizer_rate = 1;
  U64 opt_coarse_channels = 1;
  U64 opt_fine_time = 32;
  U64 opt_beams = 1;

  int opt;
  while((opt = getopt(argc, argv, "hb:c:t:u:")) != -1){
    switch (opt) {
      case 'h':
        usage(argv[0]);
        return 0;
      case 'c':
        BL_INFO("Specified frequency_channels: {}", optarg);
        opt_coarse_channels = U64(atoi(optarg));
        break;
      case 't':
        BL_INFO("Specified time_samples: {}*channelizer_rate", optarg);
        opt_fine_time = U64(atoi(optarg));
        break;
      case 'u':
        BL_INFO("Specified channelizer_rate: {}", optarg);
        opt_channelizer_rate = U64(atoi(optarg));
        break;
      case 'b':
        BL_INFO("Specified number of beams: {}", optarg);
        opt_beams = U64(atoi(optarg));
        break;
      default:
        break;
    }
  }
  
  BL_INFO("Beamforming from GUPPI RAW file with the VLA Mode B Pipeline.");

  guppiraw_iterate_info_t gr_iterate = {0};

  if(guppiraw_iterate_open_stem(argv[optind], &gr_iterate)) {
    printf("'%s'...\n", argv[optind]);
    printf("Could not open: %s.%04d.raw\n", gr_iterate.stempath, gr_iterate.fileenum);
    return 1;
  }
  guppiraw_datashape_t *datashape = &gr_iterate.file_info.block_info.datashape;
  
  const U64 gr_iter_ntime = opt_fine_time*opt_channelizer_rate;
  const U64 gr_iter_nchan = opt_coarse_channels;

  ModeB<CF32>::Config config = {
    .numberOfAntennas = 1,
    .numberOfFrequencyChannels = gr_iter_nchan,
    .numberOfTimeSamples = gr_iter_ntime,
    .numberOfPolarizations = datashape->n_pol,

    .channelizerRate = opt_channelizer_rate,

    .beamformerBeams = opt_beams,

    .outputMemWidth = 8192,
    .outputMemPad = 0,

    .castBlockSize = 32,
    .channelizerBlockSize = opt_fine_time,
    .phasorsBlockSize = 32,
    .beamformerBlockSize = opt_fine_time,
  };
  const int numberOfWorkers = 2;
  Runner<ModeB<CF32>> runner = Runner<ModeB<CF32>>(numberOfWorkers, config);

  void *input_byte_buffers[numberOfWorkers];
  Vector<Device::CPU, CI8> *input_buffers[numberOfWorkers];
  Vector<Device::CPU, CF32> *phasors_buffers[numberOfWorkers];
  Vector<Device::CPU, CF32> *output_buffers[numberOfWorkers];

  for (int i = 0; i < numberOfWorkers; i++) {
    input_byte_buffers[i] = malloc(runner.getWorker().getInputSize()*2*sizeof(int8_t));
    input_buffers[i] = new Vector<Device::CPU, CI8>(input_byte_buffers[i], runner.getWorker().getInputSize());
    phasors_buffers[i] = new Vector<Device::CPU, CF32>(runner.getWorker().getPhasorsSize());
    output_buffers[i] = new Vector<Device::CPU, CF32>(runner.getWorker().getOutputSize());
  }

  U64 buffer_idx = 0, job_idx = 0;
  U64 dequeue_id, enqueue_id = 0;
  bool read_more = 1;

  long raw_read_clocks = 0;

  clock_t start = clock();

  clock_t raw_read_clock = clock();
  guppiraw_iterate_read(
    &gr_iterate,
    gr_iter_ntime,
    gr_iter_nchan,
    input_byte_buffers[buffer_idx]
  );
  raw_read_clocks += clock() - raw_read_clock;
  enqueue_id ++;

  while(enqueue_id > job_idx) {
    if (runner.enqueue(
      [&](auto& worker){
        worker.run(
          *input_buffers[buffer_idx],
          *phasors_buffers[buffer_idx],
          *output_buffers[buffer_idx]
        );
        return job_idx;
      }
    )) {
      buffer_idx = (buffer_idx + 1) % numberOfWorkers;

      if(read_more) {
        enqueue_id ++;
        raw_read_clock = clock();
        if(guppiraw_iterate_read(
          &gr_iterate,
          gr_iter_ntime,
          gr_iter_nchan,
          input_byte_buffers[buffer_idx]) <= 0
        ) {
          read_more = 0;
        }
        raw_read_clocks += clock() - raw_read_clock;
      }
    }
    
    if (runner.dequeue(&dequeue_id)) {
      job_idx++;
    }
  }

  double elapsed_sec = (double)(clock() - start) / CLOCKS_PER_SEC;

  double ingested_bytes = runner.getWorker().getInputSize()*2*sizeof(int8_t);
  ingested_bytes *= enqueue_id;

  printf("File processed in %lf s (%lf GB/s).\n", elapsed_sec, ingested_bytes / (elapsed_sec * 1e9));
  
  double guppiraw_sec = (double)(raw_read_clocks) / CLOCKS_PER_SEC;
  printf("GUPPI RAW operations: %lf s (%lf GB/s).\n", guppiraw_sec, ingested_bytes / (guppiraw_sec * 1e9));
  
  double blade_sec = elapsed_sec - guppiraw_sec;
  printf("BLADE operations: %lf s (%lf GB/s).\n", blade_sec, ingested_bytes / (blade_sec * 1e9));

  for (int i = 0; i < numberOfWorkers; i++) {
    free(input_byte_buffers[i]);
  }

  return 0;
}
